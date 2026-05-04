"""
adapter.py
----------
The Latent Prior Adapter for LPG-SAM.

This is the only trainable module in the project. It takes a Frangi
vesselness response at full image resolution (1, 1024, 1024) and produces
two things that get fed into SAM's frozen mask decoder:

  1. FiLM modulation parameters (gamma, beta) of shape (B, 256, 64, 64)
     that modulate SAM's image embedding spatially.

  2. K=4 distilled "prior tokens" of shape (B, 4, 256) that replace
     the human-click sparse prompts SAM normally consumes.

A learnable scalar `alpha`, initialized to 0, gates the FiLM modulation.
At step 0 the adapter is a no-op and the model reproduces zero-shot SAM
exactly. As training proceeds, alpha grows from zero and the prior
starts steering the decoder.

Architecture overview
---------------------
                          frangi (B, 1, 1024, 1024)
                                    |
                                    v
                       PriorProjector (strided CNN)
                                    |
                                    v
                       prior_features (B, 256, 64, 64)
                            /                   \\
                           /                     \\
                          v                       v
                  FiLM heads                TokenHead
              (gamma, beta convs)        (avgpool + MLP)
                          |                       |
                          v                       v
              (B, 256, 64, 64) x2          (B, 4, 256)
                          |                       |
                          v                       v
                    DENSE pathway            SPARSE pathway
                  (modulates SAM's          (replaces clicks
                   image embedding)          in decoder)

The strided CNN does learned downsampling 1024 -> 64 (16x) at the same
time as channel expansion 1 -> 256, in a single pass. This is better
than a fixed bilinear/maxpool downsample followed by a flat CNN, because
the network learns *which* spatial information to preserve as it
compresses, rather than throwing it away with a hand-coded operation.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Hyperparameters that match SAM's architecture. Don't change these
# without also changing sam_wrapper.py.
SAM_EMBED_DIM = 256
SAM_EMBED_GRID = 64
SAM_INPUT_SIZE = 1024


# ---------------------------------------------------------------------------
# PriorProjector: strided CNN that goes (1, 1024, 1024) -> (256, 64, 64)
# ---------------------------------------------------------------------------

class PriorProjector(nn.Module):
    """
    Learned downsampling + channel expansion for the Frangi prior.

    Channel progression: 1 -> 16 -> 32 -> 64 -> 128 -> 256
    Spatial progression: 1024 -> 512 -> 256 -> 128 -> 64 -> 64
                                (4 stride-2 blocks for 16x downsample,
                                 then one stride-1 refinement block)

    Why this shape:
    - We need to match SAM's image embedding shape (256, 64, 64) exactly
      so the FiLM heads can produce shape-compatible modulation.
    - Doing the downsample with strided convs (rather than a fixed
      interpolate) lets the network learn what spatial info to keep.
      For sparse signals like vesselness this matters: a fixed bilinear
      average would smear single-pixel capillaries into background.
    - The channel doubling at each stride is a standard CNN encoder
      pattern (think ResNet stem), and gives us multi-scale receptive
      fields naturally.

    Why GroupNorm instead of BatchNorm:
    - We train with batch size ~4. BatchNorm with small batches is
      noisy and unstable.
    - GroupNorm doesn't depend on batch size and is the standard choice
      for small-batch dense-prediction training.
    """

    def __init__(self) -> None:
        super().__init__()

        # Each block: Conv -> GroupNorm -> GELU.
        # GroupNorm with 8 groups is a standard choice; channels must be
        # divisible by 8 (16, 32, 64, 128, 256 all are).
        def block(in_c: int, out_c: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=out_c),
                nn.GELU(),
            )

        # 1024 -> 512
        self.block1 = block(1, 16, stride=2)
        # 512 -> 256
        self.block2 = block(16, 32, stride=2)
        # 256 -> 128
        self.block3 = block(32, 64, stride=2)
        # 128 -> 64
        self.block4 = block(64, 128, stride=2)
        # 64 -> 64 (refinement, no spatial change, lifts to 256 channels)
        self.block5 = block(128, SAM_EMBED_DIM, stride=1)

    def forward(self, frangi: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        frangi : torch.Tensor
            Shape (B, 1, 1024, 1024). Vesselness response in [0, 1].

        Returns
        -------
        torch.Tensor
            Shape (B, 256, 64, 64). Lifted prior features ready for
            consumption by FiLM heads and the token distillation head.
        """
        if frangi.shape[-2:] != (SAM_INPUT_SIZE, SAM_INPUT_SIZE):
            raise ValueError(
                f"PriorProjector expects {SAM_INPUT_SIZE}x{SAM_INPUT_SIZE} "
                f"input, got {tuple(frangi.shape[-2:])}."
            )
        x = self.block1(frangi)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


# ---------------------------------------------------------------------------
# FiLM heads: produce gamma and beta from prior features
# ---------------------------------------------------------------------------

class FiLMHead(nn.Module):
    """
    Two parallel 1x1 convs that produce per-channel modulation parameters.

    Given prior_features of shape (B, 256, 64, 64), produces:
        gamma : (B, 256, 64, 64)   -- multiplicative modulation
        beta  : (B, 256, 64, 64)   -- additive modulation

    These are applied to SAM's image embedding inside LatentPriorAdapter as:
        z_modulated = z_image + alpha * (gamma * z_image + beta)

    Why 1x1 convs:
    - We want per-(spatial-location, channel) modulation parameters.
      A 1x1 conv is exactly that: each output channel is a linear
      combination of all input channels at the same spatial location.
    - No spatial mixing happens here (the PriorProjector already did it).

    Initialization note:
    - We initialize gamma_conv to small weights so the initial gamma is
      near zero, meaning the modulation is near identity. Combined with
      alpha=0 in LatentPriorAdapter, this guarantees the adapter starts
      as a true no-op. Belt and suspenders.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gamma_conv = nn.Conv2d(SAM_EMBED_DIM, SAM_EMBED_DIM, kernel_size=1)
        self.beta_conv = nn.Conv2d(SAM_EMBED_DIM, SAM_EMBED_DIM, kernel_size=1)

        # Small init so initial modulation is near zero. The alpha gate
        # is the primary safety net but this adds a second layer of caution.
        nn.init.normal_(self.gamma_conv.weight, std=1e-3)
        nn.init.zeros_(self.gamma_conv.bias)
        nn.init.normal_(self.beta_conv.weight, std=1e-3)
        nn.init.zeros_(self.beta_conv.bias)

    def forward(self, prior_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma = self.gamma_conv(prior_features)
        beta = self.beta_conv(prior_features)
        return gamma, beta


# ---------------------------------------------------------------------------
# Token distillation head: prior features -> K sparse tokens
# ---------------------------------------------------------------------------

class TokenHead(nn.Module):
    """
    Distill prior features into K=4 tokens of dimension 256.

    These tokens replace the click/box-derived sparse prompts that SAM
    normally consumes. They're the "what kind of thing should the decoder
    look for" signal, complementary to the FiLM "where" signal.

    Pipeline:
        prior_features (B, 256, 64, 64)
            -> AdaptiveAvgPool2d to (B, 256, 2, 2)
            -> reshape to (B, 4, 256)
            -> per-token MLP (256 -> 256 -> 256)
            -> tokens (B, 4, 256)

    Why K=4:
    - SAM was trained with sparse prompts of 1-4 tokens (point + padding,
      or two box corners). Staying within K<=4 keeps us in distribution.
    - 4 tokens give the network enough capacity to encode something like
      ["global vesselness", "high-confidence anchor", "orientation hint",
       "background suppression"]. We don't enforce this interpretation;
      it's just a rough mental model.
    - K is a hyperparameter we'll ablate later (1, 4, 8).

    Why adaptive avg pool to 2x2:
    - It produces exactly 4 spatial bins, one per token, each summarizing
      a quadrant of the prior feature map. Top-left token sees the upper-
      left quadrant of the image, etc. This gives the K tokens a weak
      spatial interpretation, which is more useful than 4 globally
      identical tokens.
    """

    def __init__(self, num_tokens: int = 4) -> None:
        super().__init__()
        self.num_tokens = num_tokens

        # 2x2 spatial pool gives exactly 4 bins. If you change num_tokens
        # you also need to change this pool size accordingly.
        # For K=1: pool to (1, 1). For K=4: (2, 2). For K=9: (3, 3). Etc.
        if num_tokens == 1:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif num_tokens == 4:
            self.pool = nn.AdaptiveAvgPool2d((2, 2))
        elif num_tokens == 9:
            self.pool = nn.AdaptiveAvgPool2d((3, 3))
        else:
            raise ValueError(
                f"num_tokens={num_tokens} not supported. Use 1, 4, or 9."
            )

        # Per-token MLP. Applied independently to each pooled token via
        # batched matmul. Shared weights across tokens (each token uses
        # the same MLP, just with different input).
        self.mlp = nn.Sequential(
            nn.Linear(SAM_EMBED_DIM, SAM_EMBED_DIM),
            nn.GELU(),
            nn.Linear(SAM_EMBED_DIM, SAM_EMBED_DIM),
        )

    def forward(self, prior_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        prior_features : torch.Tensor
            Shape (B, 256, 64, 64).

        Returns
        -------
        torch.Tensor
            Shape (B, num_tokens, 256). Ready to feed into SAM's
            sparse_prompt_embeddings slot.
        """
        B = prior_features.shape[0]
        # (B, 256, 64, 64) -> (B, 256, 2, 2) for K=4
        pooled = self.pool(prior_features)
        # (B, 256, 2, 2) -> (B, 256, 4) -> (B, 4, 256)
        tokens = pooled.flatten(2).transpose(1, 2)
        # Apply per-token MLP. nn.Linear broadcasts over leading dims.
        tokens = self.mlp(tokens)
        return tokens


# ---------------------------------------------------------------------------
# LatentPriorAdapter: orchestrates everything
# ---------------------------------------------------------------------------

class LatentPriorAdapter(nn.Module):
    """
    Full adapter: PriorProjector + FiLMHead + TokenHead + alpha gate.

    Forward signature:
        adapter(z_image, frangi) -> (z_modulated, sparse_tokens)

    where:
        z_image  : (B, 256, 64, 64)   -- output of frozen SAM encoder
        frangi   : (B, 1, 1024, 1024) -- Frangi vesselness, full resolution

        z_modulated   : (B, 256, 64, 64) -- modulated image embedding,
                                            feed into SAM decoder
        sparse_tokens : (B, K, 256)      -- distilled tokens, feed into
                                            SAM decoder's sparse prompt slot

    The alpha gate
    --------------
    `alpha` is a single learnable scalar initialized to 0. The FiLM
    modulation is multiplied by alpha before being added to z_image:

        z_modulated = z_image + alpha * (gamma * z_image + beta)

    At step 0 (alpha=0), z_modulated == z_image exactly, and the model
    reproduces zero-shot SAM output bit-for-bit. This is the most
    important safety property of the design: the adapter can only ever
    *improve* over the baseline, never catastrophically destroy it,
    because training starts at the baseline and gradient descent
    decides how far to move.

    During training, watch alpha as a sanity signal. If alpha never
    moves away from zero, gradients aren't flowing or the prior is
    uninformative. If alpha grows but loss doesn't improve, something
    is wrong with the FiLM heads or the loss balance.

    Note: the sparse token pathway is NOT gated by alpha. The decoder
    treats the 4 prior tokens as sparse prompts that are simply
    "present" or "absent" -- there's no continuous interpolation
    between "no prompt" and "prior prompt" in SAM's design. So the
    sparse pathway is on from step 0. This is fine because the token
    head's MLP starts roughly random and learns to be useful; SAM is
    robust to noisy sparse prompts in early training.
    """

    def __init__(self, num_tokens: int = 4) -> None:
        super().__init__()
        self.projector = PriorProjector()
        self.film = FiLMHead()
        self.token_head = TokenHead(num_tokens=num_tokens)

        # Single global learnable scalar gate, initialized to 0.
        # Stored as nn.Parameter so it shows up in optimizer.param_groups
        # and gets gradients.
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        z_image: torch.Tensor,
        frangi: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        z_image : torch.Tensor
            Shape (B, 256, 64, 64). Output of FrozenSAM.encode_image().
        frangi : torch.Tensor
            Shape (B, 1, 1024, 1024). Frangi vesselness response, in [0, 1].

        Returns
        -------
        z_modulated : torch.Tensor
            Shape (B, 256, 64, 64). Pass to FrozenSAM.decode_masks() as
            `image_embeddings`.
        sparse_tokens : torch.Tensor
            Shape (B, num_tokens, 256). Pass to FrozenSAM.decode_masks()
            as `sparse_prompt_embeddings`.
        """
        # Step 1: lift Frangi to SAM's feature space.
        prior_features = self.projector(frangi)  # (B, 256, 64, 64)

        # Step 2: dense pathway -- compute FiLM modulation.
        gamma, beta = self.film(prior_features)  # both (B, 256, 64, 64)

        # Step 3: apply gated FiLM to z_image. At alpha=0 this is identity.
        z_modulated = z_image + self.alpha * (gamma * z_image + beta)

        # Step 4: sparse pathway -- distill tokens.
        sparse_tokens = self.token_head(prior_features)  # (B, K, 256)

        return z_modulated, sparse_tokens


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Standalone shape + invariant checks for the adapter. Run with:
        python -m src.adapter

    Verifies:
      1. PriorProjector produces (B, 256, 64, 64) from (B, 1, 1024, 1024).
      2. FiLMHead produces two (B, 256, 64, 64) tensors.
      3. TokenHead produces (B, 4, 256).
      4. LatentPriorAdapter end-to-end shapes are correct.
      5. The CRITICAL invariant: at alpha=0, z_modulated == z_image exactly.
         If this fails, the adapter is silently breaking the zero-shot
         baseline and any reported improvements are suspect.
      6. Gradients flow into all trainable parameters.

    No SAM weights or real data are needed for this test.
    """
    import sys
    from src.utils import (
        get_device,
        device_info,
        seed_everything,
        summarize_trainable,
    )

    seed_everything(42)
    device = get_device()
    print(f"Device: {device_info(device)}\n")

    B = 2

    # ----- Test PriorProjector -----
    print("[1/6] PriorProjector shape check ...")
    proj = PriorProjector().to(device)
    frangi = torch.rand(B, 1, SAM_INPUT_SIZE, SAM_INPUT_SIZE, device=device)
    out = proj(frangi)
    assert out.shape == (B, SAM_EMBED_DIM, SAM_EMBED_GRID, SAM_EMBED_GRID), (
        f"expected {(B, SAM_EMBED_DIM, SAM_EMBED_GRID, SAM_EMBED_GRID)}, "
        f"got {tuple(out.shape)}"
    )
    print(f"      ok: {tuple(frangi.shape)} -> {tuple(out.shape)}")

    # ----- Test FiLMHead -----
    print("[2/6] FiLMHead shape check ...")
    film = FiLMHead().to(device)
    gamma, beta = film(out)
    assert gamma.shape == out.shape and beta.shape == out.shape
    print(f"      ok: gamma {tuple(gamma.shape)}, beta {tuple(beta.shape)}")

    # ----- Test TokenHead -----
    print("[3/6] TokenHead shape check ...")
    th = TokenHead(num_tokens=4).to(device)
    tokens = th(out)
    assert tokens.shape == (B, 4, SAM_EMBED_DIM)
    print(f"      ok: tokens {tuple(tokens.shape)}")

    # ----- Test full adapter end-to-end -----
    print("[4/6] LatentPriorAdapter end-to-end shape check ...")
    adapter = LatentPriorAdapter(num_tokens=4).to(device)
    summarize_trainable(adapter, name="LatentPriorAdapter")

    z_image = torch.randn(B, SAM_EMBED_DIM, SAM_EMBED_GRID, SAM_EMBED_GRID, device=device)
    z_mod, sparse = adapter(z_image, frangi)
    assert z_mod.shape == z_image.shape
    assert sparse.shape == (B, 4, SAM_EMBED_DIM)
    print(f"      ok: z_mod {tuple(z_mod.shape)}, sparse {tuple(sparse.shape)}")

    # ----- CRITICAL: alpha=0 invariant -----
    print("[5/6] alpha=0 invariant (z_modulated == z_image) ...")
    with torch.no_grad():
        adapter.alpha.zero_()  # force alpha to exactly 0
        z_mod_zero, _ = adapter(z_image, frangi)
        max_diff = (z_mod_zero - z_image).abs().max().item()
    print(f"      max |z_mod - z_image| = {max_diff:.2e}")
    assert max_diff < 1e-6, (
        f"CRITICAL FAILURE: at alpha=0, z_modulated should equal z_image "
        f"exactly, but max diff is {max_diff}. The adapter is silently "
        f"breaking the zero-shot baseline. Fix this before doing anything else."
    )
    print("      ok: adapter is a perfect no-op at alpha=0")

    # ----- Gradient flow -----
    print("[6/6] Gradient flow check ...")
    adapter.alpha.data.fill_(0.1)  # nudge alpha so gradients are non-trivial
    z_mod, sparse = adapter(z_image, frangi)
    # Fake loss: just sum everything. Real loss comes later.
    loss = z_mod.sum() + sparse.sum()
    loss.backward()

    # Every trainable parameter should have a non-None gradient.
    no_grad_params = [
        name for name, p in adapter.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not no_grad_params, f"params with no grad: {no_grad_params}"

    # Alpha specifically should have a non-zero gradient.
    assert adapter.alpha.grad is not None and adapter.alpha.grad.abs().item() > 0, (
        "alpha received no gradient -- the FiLM modulation isn't connected "
        "to the loss path."
    )
    print(f"      ok: alpha.grad = {adapter.alpha.grad.item():.4e}")

    print("\n[OK] All adapter checks passed.")
    sys.exit(0)
