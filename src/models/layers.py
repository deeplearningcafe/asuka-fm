import enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

HAS_FLASH_ATTENTION = False
try:
    from flash_attn import flash_attn_func
    from flash_attn import __version__ as fa_version

    HAS_FLASH_ATTENTION = False
    # print(f"Using flash attention with version {fa_version}")
except ImportError:
    print("Couldn't import flash attention")
    pass


def apply_rotary_emb(x: torch.Tensor, rotary_emb: torch.Tensor) -> torch.Tensor:
    """
    Applies Rotary Position Embeddings to the input tensor.
    Args:
        x: Tensor of shape [B, SeqLen, Heads, HeadDim]
        rotary_emb: Tuple of (cos, sin) or a single tensor containing both.
                    Shape should be broadcastable to [B, SeqLen, 1, HeadDim]
    """
    # Assuming rotary_emb is concatenated [cos, sin] on the last dimension
    cos, sin = rotary_emb.chunk(2, dim=-1)

    # Expand to match the Heads dimension: [B, SeqLen, 1, HeadDim]
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)

    # Rotate half the hidden dims
    x1, x2 = x.chunk(2, dim=-1)
    x_rot = torch.cat([-x2, x1], dim=-1)

    return x * cos + x_rot * sin


class TimeEmbeddings(nn.Module):
    """
    Calculates sinusoidal embeddings and projects them.
    Matches diffusers Timesteps + TimestepEmbedding structure.
    """

    def __init__(self, sinusoidal_dim: int, output_dim: int, max_period=10000):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.output_dim = output_dim
        if sinusoidal_dim % 2 != 0:
            raise ValueError(
                f"Cannot use sinusoidal dim {sinusoidal_dim}, must be even."
            )
        half_dim = sinusoidal_dim // 2

        exponent = -math.log(max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device="cuda"
        )
        exponent = exponent / half_dim

        self.register_buffer("inv_freq", torch.exp(exponent), persistent=False)

        self.linear_1 = nn.Linear(sinusoidal_dim, output_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(output_dim, output_dim)

    def _get_sinusoidal_embeddings(self, timesteps: torch.Tensor):
        """Calculates the base sinusoidal embeddings."""
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

        emb = timesteps[:, None].float() * self.inv_freq[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        half_dim = self.sinusoidal_dim // 2
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

        if self.sinusoidal_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))

        return emb

    def forward(self, timesteps: torch.Tensor, sample: torch.Tensor):
        if timesteps.max() <= 1.001:
            timesteps = timesteps * 1000.0

        timesteps = timesteps.expand(sample.shape[0])
        sin_emb = self._get_sinusoidal_embeddings(timesteps)

        sin_emb = sin_emb.to(dtype=self.linear_1.weight.dtype)

        emb = self.linear_1(sin_emb)
        emb = self.act(emb)
        emb = self.linear_2(emb)
        return emb


class Attention(nn.Module):
    def __init__(
        self,
        in_channels,
        cross_attention_dim=None,
        num_attention_heads=8,
        qk_norm=None,
        kv_num_heads=None,
        eps: float = 1e-5,
        base_sequence_length: int = None,
        use_flash_attention=HAS_FLASH_ATTENTION,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cross_attention_dim = in_channels
        self.num_attention_heads = num_attention_heads
        if cross_attention_dim:
            self.cross_attention_dim = cross_attention_dim

        self.use_flash_attention = use_flash_attention
        self.base_sequence_length = base_sequence_length
        self.head_dim = self.in_channels // self.num_attention_heads

        # If MQA (kv_num_heads=1), kv_dim is exactly 1 * head_dim (e.g., 64)
        self.kv_num_heads = (
            kv_num_heads if kv_num_heads is not None else self.num_attention_heads
        )
        self.kv_dim = self.head_dim * self.kv_num_heads

        self.scale = self.head_dim**-0.5

        self.to_q = nn.Linear(
            in_features=self.in_channels, out_features=self.in_channels, bias=False
        )
        self.to_k = nn.Linear(
            in_features=self.cross_attention_dim,
            out_features=self.kv_dim,
            bias=False,
        )
        self.to_v = nn.Linear(
            in_features=self.cross_attention_dim,
            out_features=self.kv_dim,
            bias=False,
        )

        self.to_out = nn.Linear(
            in_features=self.in_channels, out_features=self.in_channels, bias=False
        )

        self.norm_q = None
        self.norm_k = None
        if qk_norm == "rms_norm":
            self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
            self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)

        if self.use_flash_attention:
            self.forward = self.forward_flash_attention
        else:
            self.forward = self.forward_sdpa

    def forward_sdpa(
        self, x, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None
    ):
        batch, T, C = x.shape

        # Dynamic QKV/KV linear fusion
        if encoder_hidden_states is None or encoder_hidden_states is x:
            if self.cross_attention_dim == self.in_channels:
                qkv_w = torch.cat(
                    [self.to_q.weight, self.to_k.weight, self.to_v.weight],
                    dim=0,
                )
                qkv = F.linear(x, qkv_w)
                q, k, v = torch.split(
                    qkv,
                    [self.in_channels, self.kv_dim, self.kv_dim],
                    dim=-1,
                )
            else:
                q = self.to_q(x)
                k = self.to_k(x)
                v = self.to_v(x)
        else:
            q = self.to_q(x)
            kv_w = torch.cat([self.to_k.weight, self.to_v.weight], dim=0)
            kv = F.linear(encoder_hidden_states, kv_w)
            k, v = torch.split(kv, [self.kv_dim, self.kv_dim], dim=-1)

        # reshape with multi heads
        # GQA: q[B, H*W, C] -> [B, H*W, Heads, C/Heads], k: [B, T, C] -> [B, H*W, KV_Heads, C/KV_Heads]
        q = q.view(batch, -1, self.num_attention_heads, self.head_dim)
        k = k.view(batch, -1, self.kv_num_heads, self.head_dim)
        v = v.view(batch, -1, self.kv_num_heads, self.head_dim)

        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)

        if attention_mask is not None:
            # attention_mask is [B, T_k]
            # SDPA expects mask of shape [B, 1, 1, T_k] to broadcast over heads and queries
            sdpa_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            sdpa_mask = None

        # Apply RoPE after QK-norm
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

        # Transpose for SDPA: [B, Heads, SeqLen, HeadDim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # repeat KV heads to match Q heads
        if self.num_attention_heads != self.kv_num_heads:
            n_rep = self.num_attention_heads // self.kv_num_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        softmax_scale = self.scale
        if self.base_sequence_length is not None:
            softmax_scale = (
                math.sqrt(math.log(T, self.base_sequence_length)) * self.scale
            )

        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=sdpa_mask, is_causal=False, scale=softmax_scale
        )

        # attn: [B, Heads, H*W, C/Heads] -> [B, H*W, Heads, C/Heads]
        x = x.transpose(1, 2).reshape(batch, -1, C)

        x = self.to_out(x)
        return x

    def forward_flash_attention(
        self, x, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None
    ):
        # the input is [B, T, C]
        B, T, C = x.shape
        if encoder_hidden_states is None or encoder_hidden_states is x:
            if self.cross_attention_dim == self.in_channels:
                qkv_w = torch.cat(
                    [self.to_q.weight, self.to_k.weight, self.to_v.weight],
                    dim=0,
                )
                qkv = F.linear(x, qkv_w)
                q, k, v = torch.split(
                    qkv,
                    [self.in_channels, self.kv_dim, self.kv_dim],
                    dim=-1,
                )
            else:
                q = self.to_q(x)
                k = self.to_k(x)
                v = self.to_v(x)
        else:
            q = self.to_q(x)
            kv_w = torch.cat([self.to_k.weight, self.to_v.weight], dim=0)
            kv = F.linear(encoder_hidden_states, kv_w)
            k, v = torch.split(kv, [self.kv_dim, self.kv_dim], dim=-1)

        # q [B, H*W, Heads, HeadDim], k [B, T, KV_Heads, HeadDim]
        q = q.view(B, -1, self.num_attention_heads, self.head_dim)
        k = k.view(B, -1, self.kv_num_heads, self.head_dim)
        v = v.view(B, -1, self.kv_num_heads, self.head_dim)

        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

        softmax_scale = self.scale
        if self.base_sequence_length is not None:
            softmax_scale = (
                math.sqrt(math.log(T, self.base_sequence_length)) * self.scale
            )

        x = flash_attn_func(q, k, v, causal=False, softmax_scale=softmax_scale)

        # attn [B, H*W, NH, NDIM]
        x = x.contiguous().view(B, -1, C)

        x = self.to_out(x)

        return x


class GEGLU(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        # concatenate output dims
        self.proj_in = nn.Linear(
            in_features=self.in_channels,
            out_features=self.out_channels * 2,
            bias=self.bias,
        )

    def forward(self, x):
        hidden_states, gate = self.proj_in(x).chunk(2, dim=-1)
        return hidden_states * torch.nn.functional.gelu(gate)


class SwiGLU(nn.Module):
    """
    Swish Gated Linear Unit (SwiGLU) activation module.
    """

    def __init__(self, in_channels, out_channels, bias=True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.proj_in = nn.Linear(
            in_features=self.in_channels,
            out_features=self.out_channels * 2,
            bias=self.bias,
        )

    def forward(self, x):
        hidden_states, gate = self.proj_in(x).chunk(2, dim=-1)
        return hidden_states * torch.nn.functional.silu(gate)


class Feedforward(nn.Module):
    def __init__(
        self, in_channels, expansion_ratio=4, activation_func: str = "geglu"
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        hidden_channels = int(in_channels * expansion_ratio)

        if activation_func == "geglu":
            self.geglu = GEGLU(self.in_channels, hidden_channels, bias=True)
        elif activation_func == "swiglu":
            self.geglu = SwiGLU(self.in_channels, hidden_channels, bias=True)
        else:
            raise ValueError(f"Unknown activation: {activation_func}")

        self.proj_out = nn.Linear(
            in_features=hidden_channels, out_features=self.in_channels, bias=True
        )

    def forward(self, x):
        x = self.geglu(x)
        x = self.proj_out(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        cross_attention_dim=None,
        num_attention_heads: int = 8,
        use_checkpointing: bool = True,
        disable_self_attention: bool = False,
        qk_norm: str = None,
        kv_num_heads: int = None,
        ffn_expansion_ratio: int = 4,
        norm_type: str = "layer_norm",
        activation_func: str = "geglu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cross_attention_dim = cross_attention_dim
        self.num_attention_heads = num_attention_heads
        self.use_checkpointing = use_checkpointing
        self.disable_self_attention = disable_self_attention

        def get_norm(dim):
            if norm_type == "layer_norm":
                return nn.LayerNorm(dim)
            elif norm_type == "rms_norm":
                return nn.RMSNorm(dim)
            else:
                raise ValueError(f"Unknown norm_type: {norm_type}")

        # self attn
        if not self.disable_self_attention:
            self.norm1 = get_norm(self.in_channels)
            self.attn1 = Attention(
                in_channels=self.in_channels,
                num_attention_heads=self.num_attention_heads,
                qk_norm=qk_norm,
                kv_num_heads=kv_num_heads,
            )

        # cross attn
        if cross_attention_dim is not None:
            self.norm2 = get_norm(self.in_channels)
            self.attn2 = Attention(
                in_channels=self.in_channels,
                num_attention_heads=self.num_attention_heads,
                cross_attention_dim=self.cross_attention_dim,
                qk_norm=qk_norm,
                kv_num_heads=kv_num_heads,
            )

        # feed forward
        self.norm3 = get_norm(self.in_channels)
        self.ff = Feedforward(
            self.in_channels,
            expansion_ratio=ffn_expansion_ratio,
            activation_func=activation_func,
        )

    def forward(
        self, x, encoder_hidden_states, attention_mask=None, image_rotary_emb=None
    ):
        if not self.disable_self_attention:
            x_norm = self.norm1(x)
            # attention mask if there is NO cross-attention (imgs no mask)
            self_attn_mask = attention_mask if encoder_hidden_states is None else None
            hidden_states = x + self.attn1(
                x_norm, attention_mask=self_attn_mask, image_rotary_emb=image_rotary_emb
            )
        else:
            hidden_states = x

        if encoder_hidden_states is not None:
            hidden_states_norm = self.norm2(hidden_states)
            hidden_states = hidden_states + self.attn2(
                hidden_states_norm, encoder_hidden_states, attention_mask=attention_mask
            )

        hidden_states_norm = self.norm3(hidden_states)
        if self.use_checkpointing:
            ff_out = torch.utils.checkpoint.checkpoint(
                self.ff, hidden_states_norm, use_reentrant=False
            )
            hidden_states = hidden_states + ff_out
        else:
            hidden_states = hidden_states + self.ff(hidden_states_norm)

        return hidden_states


class TransformerTextAdapter(nn.Module):
    """
    A strong transformer-based text adapter as proposed in the i1 paper.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        num_layers: int = 2,
        num_attention_heads: int = 8,
        ffn_expansion_ratio: float = 4.0,
        use_checkpointing: bool = True,
        norm_type: str = "layer_norm",
        activation_func: str = "geglu",
    ):
        super().__init__()
        self.proj_in = nn.Linear(in_channels, hidden_size)
        self.use_checkpointing = use_checkpointing

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    in_channels=hidden_size,
                    cross_attention_dim=None,  # Self-attention only
                    num_attention_heads=num_attention_heads,
                    use_checkpointing=self.use_checkpointing,
                    disable_self_attention=False,
                    qk_norm="rms_norm",
                    ffn_expansion_ratio=ffn_expansion_ratio,
                    norm_type=norm_type,
                    activation_func=activation_func,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        text_rotary_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(
                x,
                encoder_hidden_states=None,
                attention_mask=attention_mask,
                image_rotary_emb=text_rotary_emb,
            )
        return x


# based on https://github.com/zlab-princeton/i1/blob/main/torch_inference/generate.py
def _default_rope_axes_dims(head_dim: int) -> tuple[int, int, int]:
    """Splits the head dimension into 3 chunks for Text, Y, and X coordinates."""
    if head_dim % 2 != 0:
        raise ValueError("Head dimension must be even for RoPE.")
    time_dim = head_dim // 2
    if time_dim % 2 != 0:
        time_dim -= 1
    remaining = head_dim - time_dim
    row_dim = remaining // 2
    col_dim = remaining - row_dim
    if row_dim % 2 != 0:
        row_dim -= 1
        col_dim += 1
    if col_dim % 2 != 0:
        col_dim -= 1
        row_dim += 1
    if min(time_dim, row_dim, col_dim) <= 0:
        raise ValueError("Each RoPE axis must receive at least two dimensions.")
    return time_dim, row_dim, col_dim


class MultimodalRopeEmbedder(nn.Module):
    """3D RoPE for Joint Text and Image streams supporting discrete and

    continuous coordinates.
    """

    def __init__(
        self,
        axes_dims: tuple[int, int, int],
        max_text_len: int = 512,
        max_spatial_dim: int = 128,
        theta: float = 10000.0,
        use_continuous: bool = True,
    ) -> None:
        super().__init__()
        self.use_continuous = use_continuous
        axes_lens = (max_text_len, max_spatial_dim, max_spatial_dim)

        cos_tables = []
        sin_tables = []
        inv_freqs = []
        for dim, axis_len in zip(axes_dims, axes_lens):
            steps = torch.arange(0, dim, 2, dtype=torch.float32)
            base = 1.0 / (theta ** (steps / dim))
            inv_freqs.append(base)

            positions = torch.arange(axis_len, dtype=torch.float32)
            angles = positions[:, None] * base[None, :]
            cos_tables.append(angles.cos())
            sin_tables.append(angles.sin())

        self.cos_tables = nn.ParameterList(
            [nn.Parameter(t, requires_grad=False) for t in cos_tables]
        )
        self.sin_tables = nn.ParameterList(
            [nn.Parameter(t, requires_grad=False) for t in sin_tables]
        )
        self.inv_freqs = nn.ParameterList(
            [nn.Parameter(f, requires_grad=False) for f in inv_freqs]
        )

        # Method binding in __init__ prevents graph breaks during torch.compile
        if self.use_continuous:
            self.forward = self._forward_continuous
        else:
            self.forward = self._forward_discrete

    def _forward_continuous(
        self, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Continuous coordinate rotary embedding via on-the-fly math."""
        cos = []
        sin = []
        pos_float = position_ids.float()
        for axis_idx, inv_freq in enumerate(self.inv_freqs):
            pos = pos_float[:, :, axis_idx]
            # Outer product: [B, SeqLen, 1] * [1, 1, Dim // 2]
            angles = pos.unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
            cos.append(angles.cos())
            sin.append(angles.sin())
        return torch.cat(cos, dim=-1), torch.cat(sin, dim=-1)

    def _forward_discrete(
        self, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Legacy table lookup rotary embedding for integer index grids."""
        cos = []
        sin = []
        pos_long = position_ids.long()
        for axis_idx, (cos_table, sin_table) in enumerate(
            zip(self.cos_tables, self.sin_tables)
        ):
            pos = pos_long[:, :, axis_idx].clamp(0, cos_table.shape[0] - 1)
            cos.append(F.embedding(pos, cos_table))
            sin.append(F.embedding(pos, sin_table))
        return torch.cat(cos, dim=-1), torch.cat(sin, dim=-1)
