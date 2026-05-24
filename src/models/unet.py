import safetensors
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import re

HAS_FLASH_ATTENTION = False
try:
    from flash_attn import flash_attn_func
    from flash_attn import flash_attn_varlen_func
    from flash_attn import __version__ as fa_version

    HAS_FLASH_ATTENTION = True
    print(f"Using flash attention with version {fa_version}")
except ImportError:
    print("Couldn't import flash attention")
    pass


class ResnetBlock(nn.Module):
    def __init__(
        self, input_channels, output_channels, time_embeddings, num_groups, eps
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.time_embeddings = time_embeddings
        self.num_groups = num_groups
        self.eps = eps

        self.norm1 = nn.GroupNorm(
            num_groups=self.num_groups, num_channels=self.input_channels, eps=self.eps
        )
        self.conv1 = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.nonlinearity = nn.SiLU()
        self.time_emb_proj = nn.Linear(
            in_features=self.time_embeddings,
            out_features=self.output_channels,
            bias=True,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=self.num_groups, num_channels=self.output_channels, eps=self.eps
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.output_channels,
            out_channels=self.output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        use_conv_shortcut = True if input_channels != output_channels else False
        self.conv_shortcut = None
        if use_conv_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=self.output_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )

    def forward(self, x, temb):
        hidden_states = x

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        # Time embedding projection
        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)[:, :, None, None]
        hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)

        output = hidden_states + x

        return output


class UpsamplerBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward(self, x):
        # upsample_nearest_nhwc fails with large batch sizes
        scale_factor = 2.0
        x = x.contiguous()
        x = F.interpolate(x, scale_factor=scale_factor, mode="nearest")
        x = self.conv(x)

        return x


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
            emb = F.pad(emb, (0, 1))

        return emb

    def forward(self, timesteps: torch.Tensor, sample: torch.Tensor):
        timesteps = timesteps.expand(sample.shape[0])
        sin_emb = self._get_sinusoidal_embeddings(timesteps)

        sin_emb = sin_emb.to(dtype=self.linear_1.weight.dtype)

        # Project embeddings
        emb = self.linear_1(sin_emb)
        emb = self.act(emb)
        emb = self.linear_2(emb)
        return emb


class Attention(nn.Module):
    def __init__(
        self,
        input_channels,
        n_head,
        cross_attention_dim=None,
        use_flash_attention=HAS_FLASH_ATTENTION,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.n_head = n_head
        self.cross_attention_dim = input_channels
        if cross_attention_dim:
            self.cross_attention_dim = cross_attention_dim
        self.use_flash_attention = use_flash_attention

        self.to_q = nn.Linear(self.input_channels, self.input_channels, bias=False)
        self.to_k = nn.Linear(self.cross_attention_dim, self.input_channels, bias=False)
        self.to_v = nn.Linear(self.cross_attention_dim, self.input_channels, bias=False)

        self.to_out = nn.Linear(self.input_channels, self.input_channels, bias=True)

        # Assign the forward method based on the availability of FA
        if self.use_flash_attention:
            self.forward = self.forward_flash_attention
        else:
            self.forward = self.forward_sdpa

    def forward_sdpa(self, x, encoder_hidden_states=None, attention_mask=None):
        # the input is [B, C, H, W]
        B, T, C = x.shape
        q = self.to_q(x)

        encoder_hidden_states = (
            encoder_hidden_states if encoder_hidden_states is not None else x
        )
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        # q [B, H*W, C], k [B, T, C],
        q = q.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)

        if attention_mask is not None:
            # attention_mask is [B, T_k]
            # SDPA expects mask of shape [B, 1, 1, T_k] to broadcast over heads and queries
            sdpa_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            sdpa_mask = None

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=sdpa_mask, is_causal=False
        )
        # attn [B, H*W, NH, NDIM]
        x = x.transpose(1, 2).contiguous().view(B, -1, C)

        x = self.to_out(x)

        return x

    def forward_flash_attention(
        self, x, encoder_hidden_states=None, attention_mask=None
    ):
        B, T_q, C = x.shape
        q = self.to_q(x)

        encoder_hidden_states = (
            encoder_hidden_states if encoder_hidden_states is not None else x
        )
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        q = q.view(B, -1, self.n_head, C // self.n_head)
        k = k.view(B, -1, self.n_head, C // self.n_head)
        v = v.view(B, -1, self.n_head, C // self.n_head)

        if attention_mask is not None:
            # 1. Flatten Q (Q has no padding, so it's straightforward)
            q_unpad = q.flatten(0, 1)
            cu_seqlens_q = torch.arange(
                0, (B + 1) * T_q, step=T_q, dtype=torch.int32, device=q.device
            )

            # 2. Flatten K and V, dropping the padded tokens
            mask_flat = attention_mask.flatten()
            k_unpad = k.flatten(0, 1)[mask_flat]
            v_unpad = v.flatten(0, 1)[mask_flat]

            # 3. Calculate cumulative sequence lengths for K/V
            seqlens_k = attention_mask.sum(dim=1, dtype=torch.int32)
            cu_seqlens_k = F.pad(torch.cumsum(seqlens_k, dim=0), (1, 0)).to(torch.int32)

            # 4. Call varlen Flash Attention
            max_seqlen_q = T_q
            max_seqlen_k = seqlens_k.max().item()

            out_unpad = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                causal=False,
            )

            # 5. Reshape back to[B, T_q, C]
            x = out_unpad.view(B, T_q, C)
        else:
            x = flash_attn_func(q, k, v, causal=False)
            # attn [B, H*W, NH, NDIM]
            x = x.contiguous().view(B, -1, C)

        x = self.to_out(x)

        return x


class GEGLU(nn.Module):
    r"""
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        return F.gelu(gate)

    def forward(
        self,
        hidden_states,
    ):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class MLP(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.input_channels = input_channels
        self.geglu = GEGLU(input_channels, 4 * input_channels, bias=True)
        self.proj_out = nn.Linear(input_channels * 4, input_channels, bias=True)

    def forward(self, x):
        x = self.geglu(x)
        x = self.proj_out(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        cross_attention_dim,
        n_head,
        use_checkpointing: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.cross_attention_dim = cross_attention_dim
        self.use_checkpointing = use_checkpointing

        self.norm1 = nn.LayerNorm((self.input_channels,))
        self.attn1 = Attention(self.input_channels, n_head)

        self.norm2 = nn.LayerNorm((self.input_channels,))
        self.attn2 = Attention(self.input_channels, n_head, self.cross_attention_dim)

        self.norm3 = nn.LayerNorm((self.input_channels,))
        self.ff = MLP(self.input_channels)

    def forward(self, x, encoder_hidden_states, attention_mask=None):
        x_norm = self.norm1(x)
        hidden_states = x + self.attn1(x_norm)

        hidden_states_norm = self.norm2(hidden_states)
        hidden_states = hidden_states + self.attn2(
            hidden_states_norm,
            encoder_hidden_states,
            attention_mask=attention_mask,
        )

        hidden_states_norm = self.norm3(hidden_states)
        if self.use_checkpointing:
            ff_out = torch.utils.checkpoint.checkpoint(
                self.ff, hidden_states_norm, use_reentrant=True
            )
            hidden_states = hidden_states + ff_out
        else:
            hidden_states = hidden_states + self.ff(hidden_states_norm)

        return hidden_states


class AttentionBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        cross_attention_dim,
        n_head,
        num_groups,
        eps,
        use_checkpointing: bool = False,
    ):
        super().__init__()

        assert input_channels % n_head == 0
        self.input_channels = input_channels
        self.cross_attention_dim = cross_attention_dim
        self.num_groups = num_groups
        self.eps = eps

        self.norm = nn.GroupNorm(
            num_groups=self.num_groups, num_channels=self.input_channels, eps=1e-6
        )
        self.proj_in = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.input_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    input_channels,
                    cross_attention_dim,
                    n_head,
                    use_checkpointing=use_checkpointing,
                )
            ]
        )

        self.proj_out = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.input_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x, encoder_hidden_states, attention_mask=None):
        batch, _, height, width = x.shape
        res = x

        x = self.norm(x)
        x = self.proj_in(x)
        inner_dim = x.shape[1]
        x = x.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)

        for block in self.transformer_blocks:
            x = block(x, encoder_hidden_states, attention_mask=attention_mask)

        # prepare output
        x = x.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        x = self.proj_out(x)

        x = x + res

        return x


@dataclass
class UnetConfig:
    in_channels: int = 4
    out_channels: int = 4
    block_out_channels: list[int] = None
    cross_attention_dim: int = 768
    num_blocks: int = 4
    attention_head_dim: int = 8
    layers_per_block: int = 2
    norm_num_groups: int = 32
    norm_eps: int = 1e-05
    use_checkpointing: bool = False

    def __post_init__(self):
        if self.block_out_channels is None:
            self.block_out_channels = [320, 640, 1280, 1280]


class Unet(nn.Module):
    def __init__(self, config):
        """
        Downsample block input and output channels, dim = 64:
            320 320 (32)
            320 640 (16)
            640 1280 (8)
            1280 1280

        Args:
            config (_type_): _description_
        """
        super().__init__()
        self.config = config
        self.use_checkpointing = config.use_checkpointing
        block_out_channels = config.block_out_channels
        time_embed_dim = block_out_channels[-1]
        self.in_channels = config.in_channels

        # 1. Input Convolution
        self.conv_in = nn.Conv2d(
            config.in_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        # 2. Time Embedding
        self.time_embedding = TimeEmbeddings(
            sinusoidal_dim=block_out_channels[0],
            output_dim=time_embed_dim,
        )

        # down_blocks is Resnet  ->  CrossAtten -> Downsample except last block is just Resnet
        # is the AttentionBlock the one with the double the channels for the up block in the unet
        # and is the resnet the one how changes the block channels
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i in range(config.num_blocks):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == config.num_blocks - 1

            resnets = nn.ModuleList([])
            attentions = nn.ModuleList([])

            for j in range(config.layers_per_block):
                # First resnet in block handles channel changes
                res_input_channel = input_channel if j == 0 else output_channel
                resnets.append(
                    ResnetBlock(
                        res_input_channel,
                        output_channel,
                        time_embed_dim,
                        config.norm_num_groups,
                        config.norm_eps,
                    )
                )
                # Add attention blocks except for the final block
                if not is_final_block:
                    attentions.append(
                        AttentionBlock(
                            output_channel,
                            config.cross_attention_dim,
                            config.attention_head_dim,
                            config.norm_num_groups,
                            config.norm_eps,
                            use_checkpointing=self.use_checkpointing,
                        )
                    )

            down_block = nn.ModuleDict(
                {
                    "resnets": resnets,
                    "attentions": attentions,
                }
            )

            if not is_final_block:
                down_block["downsamplers"] = nn.ModuleList(
                    [
                        nn.Conv2d(
                            output_channel,
                            output_channel,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        )
                    ]
                )

            self.down_blocks.append(down_block)

        self.mid_block = nn.ModuleDict(
            {
                "resnets": nn.ModuleList(
                    [
                        ResnetBlock(
                            block_out_channels[-1],
                            block_out_channels[-1],
                            time_embed_dim,
                            config.norm_num_groups,
                            config.norm_eps,
                        ),
                        ResnetBlock(
                            block_out_channels[-1],
                            block_out_channels[-1],
                            time_embed_dim,
                            config.norm_num_groups,
                            config.norm_eps,
                        ),
                    ]
                ),
                "attentions": nn.ModuleList(
                    [
                        AttentionBlock(
                            block_out_channels[-1],
                            config.cross_attention_dim,
                            config.attention_head_dim,
                            config.norm_num_groups,
                            config.norm_eps,
                            use_checkpointing=self.use_checkpointing,
                        )
                    ]
                ),
            }
        )

        # 5. Up Blocks
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        num_layers = config.layers_per_block + 1
        for i in range(config.num_blocks):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(reversed_block_out_channels) - 1)
            ]

            is_first_block = i == 0

            resnets = nn.ModuleList([])
            attentions = nn.ModuleList([])
            for j in range(num_layers):
                res_skip_channels = (
                    input_channel if (j == num_layers - 1) else output_channel
                )
                resnet_in_channels = prev_output_channel if j == 0 else output_channel
                resnets.append(
                    ResnetBlock(
                        resnet_in_channels + res_skip_channels,
                        output_channel,
                        time_embed_dim,
                        config.norm_num_groups,
                        config.norm_eps,
                    )
                )

                # in up blocks they come after resnets
                if not is_first_block:
                    attentions.append(
                        AttentionBlock(
                            output_channel,
                            config.cross_attention_dim,
                            config.attention_head_dim,
                            config.norm_num_groups,
                            config.norm_eps,
                            use_checkpointing=self.use_checkpointing,
                        )
                    )

            up_block = nn.ModuleDict({"resnets": resnets, "attentions": attentions})

            # No upsampler needed for the last block
            if i < config.num_blocks - 1:
                up_block["upsamplers"] = nn.ModuleList(
                    [UpsamplerBlock(output_channel, output_channel)]
                )

            self.up_blocks.append(up_block)

        # 6. Output Convolution
        self.conv_norm_out = nn.GroupNorm(
            num_groups=config.norm_num_groups,
            num_channels=block_out_channels[0],
            eps=config.norm_eps,
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            block_out_channels[0], config.out_channels, kernel_size=3, padding=1
        )

    def _checkpoint(self, module, *args):
        """Helper function to apply checkpointing."""
        if self.use_checkpointing:
            return torch.utils.checkpoint.checkpoint(module, *args, use_reentrant=True)
        else:
            return module(*args)

    def forward(self, x, timestep, encoder_hidden_states, attention_mask=None):
        # 1. time
        t_emb = self.time_embedding(timestep, x)
        # 2. preprocess
        x = self.conv_in(x)

        # 3. down
        down_block_res_x = (x,)

        for i, downsample_block in enumerate(self.down_blocks):
            output_states = ()
            if i != (self.config.num_blocks - 1):
                for resnet, attention in zip(
                    downsample_block.resnets, downsample_block.attentions
                ):
                    x = self._checkpoint(resnet, x, t_emb)
                    x = attention(
                        x, encoder_hidden_states, attention_mask=attention_mask
                    )

                    output_states += (x,)

                for downsamplers in downsample_block.downsamplers:
                    x = self._checkpoint(downsamplers, x)
                output_states = output_states + (x,)

            else:
                for resnet in downsample_block.resnets:
                    x = self._checkpoint(resnet, x, t_emb)
                    output_states += (x,)

            down_block_res_x += output_states

        # 4. mid
        x = self._checkpoint(self.mid_block.resnets[0], x, t_emb)
        for (
            attention,
            resnet,
        ) in zip(self.mid_block.attentions, self.mid_block.resnets[1:]):
            x = attention(x, encoder_hidden_states, attention_mask=attention_mask)
            x = self._checkpoint(resnet, x, t_emb)

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            # gets the last 3 outputs of the downblocks (the resnets)
            res_x_tuple = down_block_res_x[-len(upsample_block.resnets) :]
            down_block_res_x = down_block_res_x[: -len(upsample_block.resnets)]
            if i == 0:
                for resnet in upsample_block.resnets:
                    res_x = res_x_tuple[-1]
                    res_x_tuple = res_x_tuple[:-1]
                    x = torch.cat([x, res_x], dim=1)

                    x = self._checkpoint(resnet, x, t_emb)

                for upsampler in upsample_block.upsamplers:
                    x = self._checkpoint(upsampler, x)

            elif i == len(self.up_blocks) - 1:
                for resnet, attention in zip(
                    upsample_block.resnets, upsample_block.attentions
                ):
                    res_x = res_x_tuple[-1]
                    res_x_tuple = res_x_tuple[:-1]
                    x = torch.cat([x, res_x], dim=1)

                    x = self._checkpoint(resnet, x, t_emb)
                    x = attention(
                        x, encoder_hidden_states, attention_mask=attention_mask
                    )

            else:
                for resnet, attention in zip(
                    upsample_block.resnets, upsample_block.attentions
                ):
                    res_x = res_x_tuple[-1]
                    res_x_tuple = res_x_tuple[:-1]
                    x = torch.cat([x, res_x], dim=1)

                    x = self._checkpoint(resnet, x, t_emb)
                    x = attention(
                        x, encoder_hidden_states, attention_mask=attention_mask
                    )

                for upsampler in upsample_block.upsamplers:
                    x = self._checkpoint(upsampler, x)

        # 6. post-process
        x = self._checkpoint(self.conv_norm_out, x)
        x = self._checkpoint(self.conv_act, x)
        x = self.conv_out(x)
        return x

    @classmethod
    def from_pretrained(cls, config, model_path, output_head_path: str = None):
        """
        Loads pretrained weights from a .safetensors file, handling
        potential differences in layer naming conventions.
        Optionally patches the output head with weights from output_head_path.
        """
        print(f"Loading weights from {model_path}")

        model = cls(config)
        sd_custom = model.state_dict()
        sd_custom_keys = set(sd_custom.keys())

        sd_pretrained = {}
        with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                sd_pretrained[k] = f.get_tensor(k)

        mapped_sd = {}
        loaded_keys = set()

        for k_pretrained, v_pretrained in sd_pretrained.items():
            k_custom = k_pretrained

            # 1. Attention block's output projection: to_out.0 -> to_out
            k_custom = re.sub(
                r"\.attn([12])\.to_out\.0\.", r".attn\1.to_out.", k_custom
            )

            # 2. FeedForward/MLP block: ff.net.0.proj -> ff.geglu.proj
            k_custom = k_custom.replace("ff.net.0.proj", "ff.geglu.proj")

            # 3. FeedForward/MLP block: ff.net.2 -> ff.proj_out
            k_custom = k_custom.replace("ff.net.2", "ff.proj_out")

            # 4. Downsamplers/Upsamplers nesting: downsamplers.0 / upsamplers.0 -> downsamplers / upsamplers
            k_custom = re.sub(r"downsamplers\.0\.conv\.", r"downsamplers.0.", k_custom)

            if k_custom in sd_custom:
                if sd_custom[k_custom].shape != v_pretrained.shape:
                    print(
                        f"Warning: Shape mismatch for key {k_pretrained} "
                        f"(-> {k_custom}). "
                        f"Pretrained: {v_pretrained.shape}, "
                        f"Custom: {sd_custom[k_custom].shape}. Skipping."
                    )
                else:
                    mapped_sd[k_custom] = v_pretrained
                    loaded_keys.add(k_custom)
            else:
                print(
                    f"Warning: Pretrained key {k_pretrained} "
                    f"(mapped to {k_custom}) not found in custom model. "
                    "Skipping."
                )

        missing_keys = sd_custom_keys - loaded_keys
        if missing_keys:
            print(
                "\nWarning: The following keys were missing in the "
                "pretrained weights and were left initialized:"
            )
            for key in sorted(list(missing_keys)):
                print(f"  {key}")

        incompatible_keys = model.load_state_dict(mapped_sd, strict=False)
        if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
            print("\nLoad State Dict Report:")
            print(f"  Missing keys: {incompatible_keys.missing_keys}")
            print(f"  Unexpected keys: {incompatible_keys.unexpected_keys}")

        if output_head_path:
            print(f"Patching output head from: {output_head_path}")

            # output head was saved using this model's keys
            head_sd = {}
            with safetensors.safe_open(
                output_head_path, framework="pt", device="cpu"
            ) as f:
                for k in f.keys():
                    head_sd[k] = f.get_tensor(k)

            # Load only the head weights, strict=False ensures we don't need
            # the rest of the UNet weights in this file.
            head_incompatible = model.load_state_dict(head_sd, strict=False)

            print("Output head patched successfully.")
            if head_incompatible.unexpected_keys:
                print(
                    f"Warning: Unexpected keys in output head file: "
                    f"{head_incompatible.unexpected_keys}"
                )

        print("Weights loaded successfully.")
        return model
