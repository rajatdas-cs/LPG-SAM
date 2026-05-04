"""
sam_wrapper.py
--------------
Wraps MedSAM (or vanilla SAM) ViT-B so that:

1. The image encoder and mask decoder are FROZEN (no parameter updates,
   AND no autograd activation storage on the encoder side via no_grad).
2. We can feed external `dense_prompt_embeddings` and `sparse_prompt_embeddings`
   directly into the decoder, bypassing SAM's normal PromptEncoder entirely.
   This is what makes the project zero-click: clicks/boxes are replaced by
   adapter-derived embeddings.
3. The forward pass returns mask logits at SAM's native decoder resolution
   (256 x 256). Upsampling to 1024 x 1024 happens in the loss / eval code.

MedSAM notes
------------
- Architecturally identical to SAM ViT-B. Only the checkpoint weights differ.
- MedSAM expects per-image min-max normalization to [0, 1]. Vanilla SAM uses
  ImageNet mean/std. The dataset module is responsible for picking the right
  normalization; this wrapper just runs the encoder on whatever it receives.
- MedSAM was trained primarily with BOX prompts. Our zero-shot baseline
  (in eval.py) should use a full-image bounding box, not a center point.

Why we need a wrapper at all
----------------------------
The vanilla SAM API exposes `SamPredictor` which is convenient for inference
but inflexible for training: it hides the prompt encoder, manages its own
device placement, and doesn't return tensors that play nicely with autograd.
For training the adapter we need raw, differentiable access to:
    encoder(image)               -> image embeddings
    decoder(image_emb, sparse, dense, ...) -> mask logits
This wrapper gives us exactly that and nothing more.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from segment_anything import sam_model_registry

from src.utils import freeze_module, summarize_trainable


# SAM ViT-B operates on a fixed 1024 x 1024 input. The encoder downsamples
# by 16x via patch embedding, producing a 64 x 64 grid of 256-dimensional
# feature vectors. These constants are baked into SAM's architecture and
# must not be changed.
SAM_INPUT_SIZE = 1024
SAM_EMBED_DIM = 256
SAM_EMBED_GRID = 64  # 1024 / 16
SAM_DECODER_OUTPUT = 256  # decoder produces masks at 256 x 256


class FrozenSAM(nn.Module):
    """
    Frozen-backbone wrapper around SAM / MedSAM ViT-B.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to the .pth weights file (MedSAM or vanilla SAM ViT-B).
    model_type : str
        SAM registry key. Always "vit_b" for this project.

    Attributes
    ----------
    sam : segment_anything.modeling.Sam
        The underlying SAM model (frozen).
    image_encoder : nn.Module
        Frozen ViT-B image encoder. Wrapped in no_grad in `encode_image`.
    mask_decoder : nn.Module
        Frozen mask decoder. Frozen for parameter updates, but gradients
        still FLOW through it during backward (this is what lets the
        adapter receive a learning signal from the loss).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_type: str = "vit_b",
    ) -> None:
        super().__init__()

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found at: {checkpoint_path}\n"
                f"Download MedSAM ViT-B from the MedSAM repo "
                f"(https://github.com/bowang-lab/MedSAM) and place it here."
            )

        # Load the full SAM model. This pulls in the image encoder,
        # prompt encoder, and mask decoder. We keep the prompt encoder
        # accessible so we can call its `get_dense_pe()` method (used
        # internally by the decoder for positional encodings) but we
        # never run the prompt encoder's normal forward path.
        self.sam = sam_model_registry[model_type](checkpoint=None)
        state_dict = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        # Some MedSAM checkpoints wrap the state dict in a "model" key;
        # handle both cases.
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        self.sam.load_state_dict(state_dict, strict=True)
        

        # Freeze every parameter in SAM. The adapter is the only trainable
        # piece in this project.
        freeze_module(self.sam)

        # Convenient aliases. These are the two sub-modules we actually
        # interact with from the outside.
        self.image_encoder = self.sam.image_encoder
        self.mask_decoder = self.sam.mask_decoder

        # We keep a handle to the prompt encoder ONLY to access:
        #   - get_dense_pe()  -> positional encoding the decoder needs
        #   - no_mask_embed   -> default "no dense prompt" embedding (256-dim)
        # We never call its forward() because we replace it entirely.
        self._prompt_encoder = self.sam.prompt_encoder

    # ------------------------------------------------------------------
    # Image encoding (frozen, no_grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run the frozen image encoder.

        Parameters
        ----------
        image : torch.Tensor
            Shape (B, 3, 1024, 1024). Already normalized per the chosen
            convention (MedSAM: per-image min-max to [0, 1]).

        Returns
        -------
        torch.Tensor
            Image embedding of shape (B, 256, 64, 64).

        Notes
        -----
        Wrapped in `torch.no_grad()` for two reasons:

        1. The encoder is frozen, so its gradients would be discarded
           anyway -- computing them is wasted work.
        2. More importantly, no_grad prevents PyTorch from storing the
           encoder's intermediate activations for backprop. The ViT-B
           encoder has ~89M params and the activation memory at 1024x1024
           is the dominant VRAM cost in SAM. Avoiding it is what lets us
           train on a 16 GB T4 with batch size 4.
        """
        if image.shape[-2:] != (SAM_INPUT_SIZE, SAM_INPUT_SIZE):
            raise ValueError(
                f"SAM expects {SAM_INPUT_SIZE}x{SAM_INPUT_SIZE} input, "
                f"got {tuple(image.shape[-2:])}. Resize in dataset.py."
            )
        return self.image_encoder(image)

    # ------------------------------------------------------------------
    # Decoder (frozen weights, but gradients flow through)
    # ------------------------------------------------------------------

    def decode_masks(
        self,
        image_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the frozen mask decoder with externally supplied embeddings.

        Parameters
        ----------
        image_embeddings : torch.Tensor
            Shape (B, 256, 64, 64). This is what the adapter modulates
            via FiLM before passing in here.
        sparse_prompt_embeddings : torch.Tensor
            Shape (B, N, 256). The adapter produces N=4 distilled tokens.
            For a true zero-shot baseline you can pass an empty tensor of
            shape (B, 0, 256).
        dense_prompt_embeddings : torch.Tensor
            Shape (B, 256, 64, 64). For LPG-SAM we put zeros here because
            our prior signal goes into the image embedding via FiLM, not
            via the dense prompt slot.
        multimask_output : bool
            SAM can output 3 candidate masks for ambiguity resolution.
            For binary vessel segmentation we want a single mask, so
            False is the right default.

        Returns
        -------
        low_res_masks : torch.Tensor
            Shape (B, 1, 256, 256) when multimask_output=False.
            These are LOGITS, not probabilities. Apply sigmoid in the
            loss / metric code.
        iou_predictions : torch.Tensor
            Shape (B, 1). SAM's predicted IoU for each output mask.
            We don't use this for training but return it for completeness.

        Notes
        -----
        Although the decoder's parameters are frozen (requires_grad=False),
        gradients still flow through it during backward(). This is what
        lets the loss signal reach the adapter. "Frozen" only means
        weights don't update -- the computational graph is still built.
        """
        # SAM's decoder is a per-image API: it expects image_embeddings of
        # shape (1, 256, 64, 64) and uses the batch dim for multiple prompts
        # on that one image. Passing a full batch of B images causes an
        # internal repeat_interleave that blows the batch to B², breaking the
        # dense prompt addition. We loop over the batch here so callers can
        # treat this as a normal batched forward.
        image_pe = self._prompt_encoder.get_dense_pe()  # (1, 256, 64, 64)

        masks_list, iou_list = [], []
        B = image_embeddings.shape[0]
        for i in range(B):
            m, iou = self.mask_decoder(
                image_embeddings=image_embeddings[i : i + 1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings[i : i + 1],
                dense_prompt_embeddings=dense_prompt_embeddings[i : i + 1],
                multimask_output=multimask_output,
            )
            masks_list.append(m)
            iou_list.append(iou)

        low_res_masks = torch.cat(masks_list, dim=0)    # (B, 1, 256, 256)
        iou_predictions = torch.cat(iou_list, dim=0)   # (B, 1)
        return low_res_masks, iou_predictions

    # ------------------------------------------------------------------
    # Convenience: zero-prompt embeddings for the dense slot
    # ------------------------------------------------------------------

    def zero_dense_prompt(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Build a "no dense prompt" tensor of shape (B, 256, 64, 64).

        SAM's prompt encoder normally produces this when no mask prompt
        is provided -- it's a learned embedding broadcast to the full grid.
        We use the same value so the decoder receives input it was trained
        to expect, and we route our own prior signal through the image
        embedding (via FiLM) instead.

        Returns
        -------
        torch.Tensor
            Shape (B, 256, 64, 64), same dtype as model weights.
        """
        # `no_mask_embed` is a learned (1, 256) embedding inside the
        # prompt encoder. We broadcast it to the full 64 x 64 grid,
        # which is what SAM does internally when no mask is provided.
        no_mask_embed = self._prompt_encoder.no_mask_embed.weight  # (1, 256)
        dense = no_mask_embed.reshape(1, SAM_EMBED_DIM, 1, 1).expand(
            batch_size, -1, SAM_EMBED_GRID, SAM_EMBED_GRID
        )
        return dense.to(device)

    def empty_sparse_prompt(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Build an empty sparse prompt tensor of shape (B, 0, 256).

        Used by the zero-shot baseline to feed the decoder no clicks at all.
        The adapter replaces this with its own (B, 4, 256) distilled tokens.
        """
        return torch.zeros(batch_size, 0, SAM_EMBED_DIM, device=device)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Quick sanity check. Run with:
        python -m src.sam_wrapper
    Requires the MedSAM checkpoint at ./checkpoints/medsam_vit_b.pth.
    """
    from src.utils import get_device, device_info, seed_everything

    seed_everything(42)
    device = get_device()
    print(f"Device: {device_info(device)}")

    ckpt = Path("checkpoints/medsam_vit_b.pth")
    print(f"Loading SAM from {ckpt} ...")
    model = FrozenSAM(checkpoint_path=ckpt).to(device)
    summarize_trainable(model, name="FrozenSAM")
    # Expected output: trainable should be ~0 (all frozen).

    # Dummy forward pass with random data.
    B = 1
    image = torch.rand(B, 3, SAM_INPUT_SIZE, SAM_INPUT_SIZE, device=device)
    print(f"\nRunning encoder on dummy image of shape {tuple(image.shape)} ...")
    with torch.no_grad():
        z_image = model.encode_image(image)
    print(f"  image embedding: {tuple(z_image.shape)}")
    assert z_image.shape == (B, SAM_EMBED_DIM, SAM_EMBED_GRID, SAM_EMBED_GRID)

    print("\nRunning decoder with empty sparse + default dense prompt ...")
    dense = model.zero_dense_prompt(B, device)
    sparse = model.empty_sparse_prompt(B, device)
    with torch.no_grad():
        masks, iou = model.decode_masks(z_image, sparse, dense)
    print(f"  mask logits: {tuple(masks.shape)}")
    print(f"  iou pred:    {tuple(iou.shape)}")
    assert masks.shape == (B, 1, SAM_DECODER_OUTPUT, SAM_DECODER_OUTPUT)

    print("\n[OK] FrozenSAM wrapper is working correctly.")
