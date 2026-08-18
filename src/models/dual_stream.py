import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.models.layers import (
    TimeEmbeddings,
    TransformerTextAdapter,
    MultimodalRopeEmbedder,
    _default_rope_axes_dims,
)


def _apply_multimodal_rope(
    x: torch.Tensor,
    freqs: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    if freqs is None:
        return x
    cos, sin = freqs
    orig_dtype = x.dtype
    sh = x.shape

    # Real-valued elementwise math (fuses cleanly in TorchInductor)
    x_pair = x.float().view(*sh[:-1], -1, 2)
    x0 = x_pair[..., 0]
    x1 = x_pair[..., 1]

    cos = cos.unsqueeze(2).float()
    sin = sin.unsqueeze(2).float()

    out0 = (x0 * cos - x1 * sin).unsqueeze(-1)
    out1 = (x0 * sin + x1 * cos).unsqueeze(-1)

    out = torch.cat([out0, out1], dim=-1).reshape(sh)
    return out.to(orig_dtype)


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, hidden_features: int) -> None:
        super().__init__()
        self.w12 = nn.Linear(hidden_size, 2 * hidden_features)
        self.w3 = nn.Linear(hidden_features, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class MMDiTAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv_image = nn.Linear(hidden_size, 3 * hidden_size)
        self.qkv_text = nn.Linear(hidden_size, 3 * hidden_size)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=eps)

        self.proj_image = nn.Linear(hidden_size, hidden_size)
        self.proj_text = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        image_freqs: tuple[torch.Tensor, torch.Tensor],
        text_freqs: tuple[torch.Tensor, torch.Tensor],
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, image_len, _ = image_tokens.shape
        text_len = text_tokens.shape[1]

        def project(linear: nn.Linear, x: torch.Tensor):
            qkv = linear(x).reshape(bsz, x.shape[1], 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            return q, k, v

        q_image, k_image, v_image = project(self.qkv_image, image_tokens)
        q_text, k_text, v_text = project(self.qkv_text, text_tokens)

        q_image, k_image = self.q_norm(q_image), self.k_norm(k_image)
        q_text, k_text = self.q_norm(q_text), self.k_norm(k_text)

        # Apply 3D RoPE
        q_image = _apply_multimodal_rope(q_image, image_freqs)
        k_image = _apply_multimodal_rope(k_image, image_freqs)
        q_text = _apply_multimodal_rope(q_text, text_freqs)
        k_text = _apply_multimodal_rope(k_text, text_freqs)

        # Transpose for SDPA: [B, Heads, SeqLen, HeadDim]
        q = torch.cat([q_image, q_text], dim=1).transpose(1, 2)
        k = torch.cat([k_image, k_text], dim=1).transpose(1, 2)
        v = torch.cat([v_image, v_text], dim=1).transpose(1, 2)

        image_mask = torch.ones(
            (bsz, image_len), dtype=torch.bool, device=text_tokens.device
        )
        key_mask = torch.cat([image_mask, text_mask.bool()], dim=1)
        attn_mask = key_mask[:, None, None, :]

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        out = out.transpose(1, 2).reshape(bsz, image_len + text_len, self.hidden_size)

        # Zero out masked text tokens
        out = out * key_mask[:, :, None].to(out.dtype)

        return self.proj_image(out[:, :image_len]), self.proj_text(out[:, image_len:])


class DualStreamDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        eps: float = 1e-5,
        use_skip: bool = False,
        use_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.use_skip = use_skip
        self.use_checkpointing = use_checkpointing
        if use_skip:
            self.skip_linear_image = nn.Linear(2 * hidden_size, hidden_size)
            self.skip_linear_text = nn.Linear(2 * hidden_size, hidden_size)

        # Sandwich Norm
        self.norm1 = nn.RMSNorm(hidden_size, eps=eps)
        self.norm2 = nn.RMSNorm(hidden_size, eps=eps)
        self.norm3 = nn.RMSNorm(hidden_size, eps=eps)
        self.norm4 = nn.RMSNorm(hidden_size, eps=eps)

        self.attn = MMDiTAttention(hidden_size, num_heads, eps=eps)

        hidden_features = int(2 / 3 * int(hidden_size * mlp_ratio))
        # Round up to next multiple of 128
        if hidden_features % 128 != 0:
            hidden_features = ((hidden_features + 127) // 128) * 128

        self.mlp_image = SwiGLUFFN(hidden_size, hidden_features)
        self.mlp_text = SwiGLUFFN(hidden_size, hidden_features)

    def _checkpoint(self, module, *args, **kwargs):
        if self.use_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                module, *args, **kwargs, use_reentrant=False
            )
        else:
            return module(*args, **kwargs)

    def _run_skip(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        skip_image: torch.Tensor,
        skip_text: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encapsulates skip connection calculations to checkpoint them."""
        img_out = self.skip_linear_image(torch.cat([image_tokens, skip_image], dim=-1))
        txt_out = self.skip_linear_text(torch.cat([text_tokens, skip_text], dim=-1))
        return img_out, txt_out

    def _run_post_attn(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        image_attn: torch.Tensor,
        text_attn: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Groups all post-attention calculations to discard intermediates."""
        image_tokens = image_tokens + self.norm3(image_attn)
        text_tokens = text_tokens + self.norm3(text_attn)

        # inner sandwich norms (norm2 -> FFN -> norm4)
        image_mlp = self.norm4(self.mlp_image(self.norm2(image_tokens)))
        text_mlp = self.norm4(self.mlp_text(self.norm2(text_tokens)))

        image_tokens = image_tokens + image_mlp
        text_tokens = text_tokens + text_mlp

        # use token mask out of checkpoint as no grad tensors ignored
        return image_tokens, text_tokens

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        image_freqs: tuple[torch.Tensor, torch.Tensor],
        text_freqs: tuple[torch.Tensor, torch.Tensor],
        text_mask: torch.Tensor,
        skip: tuple[torch.Tensor, torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_skip and skip is not None:
            image_tokens, text_tokens = self._checkpoint(
                self._run_skip,
                image_tokens,
                text_tokens,
                skip[0],
                skip[1],
            )

        image_attn, text_attn = self.attn(
            self.norm1(image_tokens),
            self.norm1(text_tokens),
            image_freqs,
            text_freqs,
            text_mask,
        )

        image_tokens, text_tokens = self._checkpoint(
            self._run_post_attn,
            image_tokens,
            text_tokens,
            image_attn,
            text_attn,
        )

        if text_mask is not None:
            text_tokens = text_tokens * text_mask[:, :, None].to(text_tokens.dtype)
        return image_tokens, text_tokens


class DualStreamDiT(nn.Module):
    """
    Lightweight Dual-Stream DiT based on the i1 paper.
    Uses Time Token prepending instead of AdaLN to support Flow Matching efficiently.
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
        use_rope_text_adapter: bool = False,
        norm_type: str = "layer_norm",
        activation_func: str = "geglu",
        skip_checkpointing_layers: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.use_checkpointing = use_checkpointing
        self.skip_checkpointing_layers = skip_checkpointing_layers
        # in the original i1 paper the don't use it
        self.use_rope_text_adapter = use_rope_text_adapter

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

        # 5. Dual Stream Blocks (with Long Skip Connections)
        num_in_blocks = depth // 2
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
                for i in range(num_in_blocks)
            ]
        )
        current_layer_idx += num_in_blocks

        self.mid_block = DualStreamDiTBlock(
            hidden_size,
            num_heads,
            mlp_ratio,
            eps=eps,
            use_checkpointing=should_checkpoint(current_layer_idx),
        )
        current_layer_idx += 1

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
                for i in range(1, num_in_blocks + 1)
            ]
        )
        current_layer_idx += num_in_blocks

        self.norm_final = nn.RMSNorm(hidden_size, eps=eps)
        self.proj_out = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

        self._zero_initialize_output()

    def _zero_initialize_output(self):
        """Crucial for diffusion/flow matching: start by predicting zero velocity/noise."""
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def _build_position_ids(
        self,
        text_mask: torch.Tensor,
        h: int,
        w: int,
        pos_map: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, text_len = text_mask.shape
        device = text_mask.device

        caption_positions = torch.arange(text_len, dtype=torch.long, device=device)[
            None
        ].expand(bsz, text_len)
        caption_positions = torch.where(
            text_mask.bool(),
            caption_positions,
            torch.zeros_like(caption_positions),
        )
        zeros = torch.zeros_like(caption_positions)
        caption_ids = torch.stack((caption_positions, zeros, zeros), dim=-1)

        num_image_tokens = h * w
        text_lengths = text_mask.sum(dim=1, dtype=torch.float32)
        image_time = text_lengths[:, None].expand(bsz, num_image_tokens)

        if pos_map is not None:
            # pos_map shape: [B, num_image_tokens, 2] continuous coordinates
            if pos_map.ndim == 4:
                pos_map = pos_map.flatten(1, 2)
            pos_map = pos_map.to(device, dtype=torch.float32)
            row_coords = pos_map[..., 0]
            col_coords = pos_map[..., 1]
        else:
            # Vectorized meshgrid construction without repeat_interleave
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing="ij",
            )
            row_coords = grid_y.reshape(-1)[None].expand(bsz, num_image_tokens)
            col_coords = grid_x.reshape(-1)[None].expand(bsz, num_image_tokens)

        image_ids = torch.stack((image_time, row_coords, col_coords), dim=-1)

        return caption_ids, image_ids

    def get_params(
        self,
    ) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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

        # 1. Patchify Image
        image_tokens = self.x_embedder(x).flatten(2).transpose(1, 2)  # [B, H*W, C]

        # 2. Prepare Time Token
        t_emb = self.time_embedding(t, x)
        time_token = self.time_token_proj(t_emb).unsqueeze(1)  # [B, 1, C]

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

        image_tokens, text_tokens = self.mid_block(
            image_tokens,
            text_tokens,
            image_freqs,
            text_freqs,
            text_mask,
        )

        for block in self.out_blocks:
            skip_tensors = skips.pop()
            image_tokens, text_tokens = block(
                image_tokens,
                text_tokens,
                image_freqs,
                text_freqs,
                text_mask,
                skip=skip_tensors,
            )

        tokens = self.proj_out(self.norm_final(image_tokens))  # [B, H*W, p*p*C_out]

        tokens = tokens.reshape(bsz, h_patches, w_patches, p, p, self.out_channels)
        tokens = tokens.permute(0, 5, 1, 3, 2, 4).reshape(bsz, self.out_channels, H, W)

        return tokens
