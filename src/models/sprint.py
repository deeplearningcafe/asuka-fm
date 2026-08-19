from typing import Optional
import torch
import torch.nn as nn
from src.models.layers import (
    TimeEmbeddings,
    TransformerTextAdapter,
)
from src.models.dual_stream import (
    DualStreamDiT,
    DualStreamDiTBlock,
    MultimodalRopeEmbedder,
    _default_rope_axes_dims,
)


def structured_token_drop(x, h_patches, w_patches, n=2, k=1):
    """
    Structured group-wise subsampling (Section 3.5 in SPRINT paper).
    Torch.compile friendly without graph breaks.
    """
    B, N, D = x.shape
    device = x.device

    if (h_patches % n != 0) or (w_patches % n != 0) or (h_patches * w_patches != N):
        drop_ratio = 1.0 - (k / (n * n))
        return random_token_drop(x, drop_ratio)

    h_groups, w_groups = h_patches // n, w_patches // n
    num_groups, group_size = h_groups * w_groups, n * n

    noise = torch.rand(B, num_groups, group_size, device=device)
    ids_group_keep = torch.argsort(noise, dim=-1)[:, :, :k]

    group_idx_h = (
        torch.arange(h_groups, device=device)
        .view(1, h_groups, 1, 1)
        .expand(B, -1, w_groups, k)
    )
    group_idx_w = (
        torch.arange(w_groups, device=device)
        .view(1, 1, w_groups, 1)
        .expand(B, h_groups, -1, k)
    )

    local_h = (ids_group_keep // n).view(B, h_groups, w_groups, k)
    local_w = (ids_group_keep % n).view(B, h_groups, w_groups, k)

    global_h = group_idx_h * n + local_h
    global_w = group_idx_w * n + local_w

    global_indices = global_h * w_patches + global_w
    ids_keep = global_indices.view(B, -1)
    K = ids_keep.shape[1]

    # Mask creation and compile-friendly complement index calculation
    mask_keep = torch.zeros(B, N, dtype=torch.bool, device=device)
    mask_keep.scatter_(
        dim=1,
        index=ids_keep,
        src=torch.ones_like(ids_keep, dtype=torch.bool),
    )

    mask_drop = (~mask_keep).float()
    original_indices = torch.arange(N, device=device).unsqueeze(0)
    drop_priority = mask_drop * N + original_indices.float() / (N + 1)
    ids_drop = torch.argsort(drop_priority, dim=1, descending=True)[:, : N - K]

    ids_shuffle = torch.cat([ids_keep, ids_drop], dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

    return x_masked, ids_keep, ids_restore


def get_token_drop_indices(x, h_patches, w_patches, drop_ratio):
    """
    Unified entrypoint to choose structured dropping based on drop ratios.
    """
    if drop_ratio == 0.75:
        return structured_token_drop(x, h_patches, w_patches, n=2, k=1)
    elif drop_ratio == 0.50:
        return structured_token_drop(x, h_patches, w_patches, n=2, k=2)
    else:
        return random_token_drop(x, drop_ratio)


def random_token_drop(x, drop_ratio):
    """
    Randomly drops tokens from the sequence according to the specified ratio.
    """
    B, N, D = x.shape
    K = int(round(N * (1 - drop_ratio)))
    K = max(1, min(K, N - 1))

    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :K]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
    return x_masked, ids_keep, ids_restore


def restore_full_sequence(x_masked, ids_restore, mask_token):
    """
    Reconstructs the full sequence by padding dropped indices with mask tokens.
    """
    B, K, D = x_masked.shape
    N = ids_restore.shape[1]

    mask_tokens = mask_token.expand(B, N - K, D)
    x_concat = torch.cat([x_masked, mask_tokens], dim=1)

    index = ids_restore.unsqueeze(-1).expand(-1, -1, D)
    x_full = torch.gather(x_concat, dim=1, index=index)
    return x_full


class SprintDualStreamDiT(DualStreamDiT):
    """
    Dual-Stream DiT with custom SPRINT-aligned encoder, middle,
    and decoder layers.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        patch_size: int = 2,
        hidden_size: int = 768,
        depth: int = 16,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        text_embed_dim: int = 768,
        use_checkpointing: bool = True,
        eps: float = 1e-5,
        encoder_depth: int = 2,
        decoder_depth: int = 2,
        drop_ratio: float = 0.75,
        drop_target: str = "image",
        residual_type: str = "concat_linear",
        cfg_mask_prob: float = 0.1,
        use_rope_text_adapter: bool = False,
        norm_type: str = "layer_norm",
        activation_func: str = "geglu",
        skip_checkpointing_layers: int = 0,
        use_random_drop: bool = False,
    ):
        # Prevent base constructor block initialization
        nn.Module.__init__(self)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.use_checkpointing = use_checkpointing
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.middle_depth = depth - encoder_depth - decoder_depth
        self.drop_ratio = drop_ratio
        self.drop_target = drop_target
        self.residual_type = residual_type
        self.cfg_mask_prob = cfg_mask_prob
        self.use_rope_text_adapter = use_rope_text_adapter
        self.skip_checkpointing_layers = skip_checkpointing_layers
        self.use_random_drop = use_random_drop

        def should_checkpoint(layer_idx: int) -> bool:
            return self.use_checkpointing and (
                layer_idx >= self.skip_checkpointing_layers
            )

        current_layer_idx = 0

        # 1. Image Embedder
        self.x_embedder = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

        # 2. Time Embedder (Used as a prepended token)
        self.time_embedding = TimeEmbeddings(sinusoidal_dim=256, output_dim=hidden_size)
        self.time_token_proj = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )

        # 3. Transformer Text Adapter
        text_adapter_layers = 2
        self.text_adapter = TransformerTextAdapter(
            in_channels=text_embed_dim,
            hidden_size=hidden_size,
            num_layers=text_adapter_layers,
            num_attention_heads=num_heads,
            ffn_expansion_ratio=mlp_ratio,
            use_checkpointing=should_checkpoint(current_layer_idx),
            norm_type=norm_type,
            activation_func=activation_func,
        )
        current_layer_idx += text_adapter_layers

        # 4. 3D RoPE
        head_dim = hidden_size // num_heads
        axes_dims = _default_rope_axes_dims(head_dim)
        self.rope_embedder = MultimodalRopeEmbedder(axes_dims)

        # Encoder blocks
        in_start_idx = current_layer_idx
        self.in_blocks = nn.ModuleList(
            [
                DualStreamDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    eps=eps,
                    use_checkpointing=should_checkpoint(in_start_idx + i),
                )
                for i in range(self.encoder_depth)
            ]
        )
        current_layer_idx += self.encoder_depth

        # Middle blocks
        # long_skip dual stream has a mid block so layers+1
        mid_start_idx = current_layer_idx
        self.mid_blocks = nn.ModuleList(
            [
                DualStreamDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    eps=eps,
                    use_checkpointing=should_checkpoint(mid_start_idx + i),
                )
                for i in range(self.middle_depth)
            ]
        )
        current_layer_idx += self.middle_depth

        # Decoder blocks
        out_start_idx = current_layer_idx
        self.out_blocks = nn.ModuleList(
            [
                DualStreamDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    eps=eps,
                    use_skip=True,
                    use_checkpointing=should_checkpoint(out_start_idx + i),
                )
                for i in range(self.decoder_depth)
            ]
        )
        current_layer_idx += self.decoder_depth

        self.norm_final = nn.RMSNorm(hidden_size, eps=eps)
        self.proj_out = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

        # torch compile squeezes during bw so remove dims
        self.mask_token_image = nn.Parameter(torch.zeros(self.hidden_size))
        self.mask_token_text = nn.Parameter(torch.zeros(self.hidden_size))
        torch.nn.init.normal_(self.mask_token_image, std=0.02)
        torch.nn.init.normal_(self.mask_token_text, std=0.02)

        if self.residual_type == "concat_linear":
            self.renoise_linear_image = nn.Linear(
                self.hidden_size * 2, self.hidden_size
            )
            self.renoise_linear_text = nn.Linear(self.hidden_size * 2, self.hidden_size)
            torch.nn.init.xavier_uniform_(self.renoise_linear_image.weight)
            torch.nn.init.xavier_uniform_(self.renoise_linear_text.weight)
            nn.init.zeros_(self.renoise_linear_image.bias)
            nn.init.zeros_(self.renoise_linear_text.bias)

        self._zero_initialize_output()

    def _drop_tokens(self, tokens, freqs, h_patches=None, w_patches=None):
        if not self.use_random_drop and h_patches is not None and w_patches is not None:
            (
                tokens_sparse,
                ids_keep,
                ids_restore,
            ) = get_token_drop_indices(tokens, h_patches, w_patches, self.drop_ratio)
        else:
            (
                tokens_sparse,
                ids_keep,
                ids_restore,
            ) = random_token_drop(tokens, self.drop_ratio)

        cos, sin = freqs
        D_half = cos.shape[-1]
        cos_sparse = torch.gather(
            cos,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D_half),
        )
        sin_sparse = torch.gather(
            sin,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D_half),
        )
        freqs_sparse = (cos_sparse, sin_sparse)
        return tokens_sparse, freqs_sparse, ids_keep, ids_restore

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        pos_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, _, H, W = x.shape
        p = self.patch_size
        h_patches, w_patches = H // p, W // p

        image_tokens = self.x_embedder(x).flatten(2).transpose(1, 2)

        t_emb = self.time_embedding(t, x)
        time_token = self.time_token_proj(t_emb).unsqueeze(1)

        time_mask = torch.ones((bsz, 1), dtype=torch.bool, device=x.device)
        text_mask = torch.cat([time_mask, attention_mask.bool()], dim=1)

        # 3D RoPE Frequencies
        text_pos_ids, image_pos_ids = self._build_position_ids(
            text_mask, h_patches, w_patches, pos_map=pos_map
        )
        all_pos_ids = torch.cat([text_pos_ids, image_pos_ids], dim=1)
        cos, sin = self.rope_embedder(all_pos_ids)

        text_rotary_emb = None
        # not used in orig i1 paper
        if self.use_rope_text_adapter:
            text_len = encoder_hidden_states.shape[1]
            # Expand cos and sin to full HeadDim for apply_rotary_emb
            cos_full = torch.cat([cos, cos], dim=-1)
            sin_full = torch.cat([sin, sin], dim=-1)
            rotary_emb = torch.cat([cos_full, sin_full], dim=-1)
            # first token is time step
            text_rotary_emb = rotary_emb[:, 1 : 1 + text_len]

        # 3. Prepare Text Tokens
        text_tokens = self.text_adapter(
            encoder_hidden_states,
            attention_mask=attention_mask,
            text_rotary_emb=text_rotary_emb,
        )
        text_tokens = torch.cat([time_token, text_tokens], dim=1)

        seq_text = text_tokens.shape[1]
        text_freqs = (cos[:, :seq_text], sin[:, :seq_text])
        image_freqs = (cos[:, seq_text:], sin[:, seq_text:])

        skips = []
        for block in self.in_blocks:
            image_tokens, text_tokens = block(
                image_tokens,
                text_tokens,
                image_freqs,
                text_freqs,
                text_mask,
            )
            skips.append((image_tokens, text_tokens))

        mask_token_image_3d = self.mask_token_image.view(1, 1, -1)
        mask_token_text_3d = self.mask_token_text.view(1, 1, -1)

        image_tokens_clone = image_tokens.clone()
        text_tokens_clone = text_tokens.clone()
        should_drop = self.training and (self.drop_ratio > 0.0)

        # Drop tokens based on targets
        if should_drop:
            if self.drop_target in ["image", "both"]:
                (
                    image_tokens_sparse,
                    image_freqs_sparse,
                    image_ids_keep,
                    image_ids_restore,
                ) = self._drop_tokens(image_tokens, image_freqs, h_patches, w_patches)
            else:
                image_tokens_sparse = image_tokens
                image_freqs_sparse = image_freqs

            if self.drop_target in ["text", "both"]:
                (
                    text_tokens_sparse,
                    text_freqs_sparse,
                    text_ids_keep,
                    text_ids_restore,
                ) = self._drop_tokens(text_tokens, text_freqs)
                text_mask_sparse = torch.gather(text_mask, dim=1, index=text_ids_keep)
            else:
                text_tokens_sparse = text_tokens
                text_freqs_sparse = text_freqs
                text_mask_sparse = text_mask
        else:
            image_tokens_sparse = image_tokens
            image_freqs_sparse = image_freqs
            text_tokens_sparse = text_tokens
            text_freqs_sparse = text_freqs
            text_mask_sparse = text_mask

        # Middle blocks
        for block in self.mid_blocks:
            image_tokens_sparse, text_tokens_sparse = block(
                image_tokens_sparse,
                text_tokens_sparse,
                image_freqs_sparse,
                text_freqs_sparse,
                text_mask_sparse,
            )

        # Restore sequences
        if should_drop:
            if self.drop_target in ["image", "both"]:
                image_tokens = restore_full_sequence(
                    image_tokens_sparse,
                    image_ids_restore,
                    mask_token_image_3d,
                )
            else:
                image_tokens = image_tokens_sparse

            if self.drop_target in ["text", "both"]:
                text_tokens = restore_full_sequence(
                    text_tokens_sparse,
                    text_ids_restore,
                    mask_token_text_3d,
                )
            else:
                text_tokens = text_tokens_sparse
        else:
            image_tokens = image_tokens_sparse
            text_tokens = text_tokens_sparse

        # SPRINT Path-drop CFG mask (training only)
        # TODO: implement pdg sampling
        if self.training and self.cfg_mask_prob > 0:
            sample_mask = (
                torch.rand(bsz, device=image_tokens.device) < self.cfg_mask_prob
            )
            if self.drop_target in ["image", "both"]:
                mask_tokens_expanded = mask_token_image_3d.expand(
                    bsz, image_tokens.shape[1], self.hidden_size
                )
                image_tokens = torch.where(
                    sample_mask.unsqueeze(1).unsqueeze(2),
                    mask_tokens_expanded,
                    image_tokens,
                )
            if self.drop_target in ["text", "both"]:
                mask_tokens_expanded = mask_token_text_3d.expand(
                    bsz, text_tokens.shape[1], self.hidden_size
                )
                text_tokens = torch.where(
                    sample_mask.unsqueeze(1).unsqueeze(2),
                    mask_tokens_expanded,
                    text_tokens,
                )

        # Residual Fusion
        if self.residual_type == "concat_linear":
            if self.drop_target in ["image", "both"]:
                image_tokens = torch.cat([image_tokens, image_tokens_clone], dim=-1)
                image_tokens = self.renoise_linear_image(image_tokens)
            if self.drop_target in ["text", "both"]:
                text_tokens = torch.cat([text_tokens, text_tokens_clone], dim=-1)
                text_tokens = self.renoise_linear_text(text_tokens)

        # Decoder blocks
        for block in self.out_blocks:
            skip_tensors = skips.pop() if skips else None
            image_tokens, text_tokens = block(
                image_tokens,
                text_tokens,
                image_freqs,
                text_freqs,
                text_mask,
                skip=skip_tensors,
            )

        tokens = self.proj_out(self.norm_final(image_tokens))

        tokens = tokens.reshape(bsz, h_patches, w_patches, p, p, self.out_channels)
        tokens = tokens.permute(0, 5, 1, 3, 2, 4).reshape(bsz, self.out_channels, H, W)

        return tokens
