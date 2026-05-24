import safetensors
import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import re
from dataclasses import dataclass, field
from typing import ClassVar


# https://www.researchgate.net/figure/ResNet-block-submodule-from-the-time-conditional-UNet-architecture_fig7_371684920
class ResnetBlock(nn.Module):
    def __init__(self, input_channels, output_channels, num_groups, eps):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
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

    def forward(self, x):
        hidden_states = x

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)

        hidden_states = hidden_states + x

        return hidden_states


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
        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984()
        scale_factor = 2.0
        if x.numel() * scale_factor > pow(2, 31) or x.shape[0] >= 64:
            x = x.contiguous()
        x = F.interpolate(x, scale_factor=scale_factor, mode="nearest")
        x = self.conv(x)

        return x


class Attention(nn.Module):
    def __init__(self, input_channels, n_head, num_groups, eps):
        super().__init__()
        self.input_channels = input_channels
        self.n_head = n_head
        self.num_groups = num_groups
        self.eps = eps

        self.group_norm = nn.GroupNorm(
            num_groups=self.num_groups, num_channels=self.input_channels, eps=self.eps
        )

        self.query = nn.Linear(self.input_channels, self.input_channels, bias=True)
        self.key = nn.Linear(self.input_channels, self.input_channels, bias=True)
        self.value = nn.Linear(self.input_channels, self.input_channels, bias=True)

        # in the safetensors it is a conv2d with kernel 1 and stride 1, proj_out
        self.proj_attn = nn.Linear(self.input_channels, self.input_channels, bias=True)

    def forward(self, x):
        # the input is [B, C, H, W]
        B, C, H, W = x.shape
        residual = x
        x = x.view(B, C, H * W)
        x = self.group_norm(x).transpose(1, 2)

        q = self.query(x)

        k = self.key(x)
        v = self.value(x)

        # q [B, H*W, C], k [B, T, C],
        q = q.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)

        x = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        # attn [B, H*W, NH, NDIM]
        x = x.transpose(1, 2).contiguous().view(B, -1, C)

        x = self.proj_attn(x)
        x = x.transpose(-1, -2).reshape(B, C, H, W)
        x = x + residual

        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        block_out_channels = config.block_out_channels

        self.conv_in = nn.Conv2d(
            config.in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i in range(config.num_blocks):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            resnets = nn.ModuleList([])
            for j in range(config.layers_per_block):
                res_input_channel = input_channel if j == 0 else output_channel
                resnets.append(
                    ResnetBlock(
                        res_input_channel,
                        output_channel,
                        config.norm_num_groups,
                        config.norm_eps,
                    )
                )
            # Create a module dictionary for the entire block
            down_block = nn.ModuleDict(
                {
                    "resnets": resnets,
                }
            )
            if i < config.num_blocks - 1:  # No upsampler needed for the last block
                down_block["downsamplers"] = nn.ModuleList(
                    [
                        nn.Conv2d(
                            output_channel,
                            output_channel,
                            kernel_size=3,
                            stride=2,
                            padding=0,
                        )
                    ]
                )

            self.down_blocks.append(down_block)

        # last layer is a resnet -> attention -> resnet
        self.mid_block = nn.ModuleDict(
            dict(
                resnets=nn.ModuleList(
                    [
                        ResnetBlock(
                            output_channel,
                            output_channel,
                            config.norm_num_groups,
                            config.norm_eps,
                        )
                        for _ in range(config.layers_per_block)
                    ]
                ),
                attentions=nn.ModuleList(
                    [
                        Attention(
                            output_channel,
                            config.attention_head_dim,
                            config.norm_num_groups,
                            config.norm_eps,
                        )
                    ]
                ),
            )
        )

        self.conv_norm_out = nn.GroupNorm(
            num_groups=config.norm_num_groups, num_channels=output_channel, eps=1e-6
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            output_channel,
            config.latent_channels * 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        x = self.conv_in(x)

        out_states = (x,)
        for i, downsample_block in enumerate(self.down_blocks):
            for resnet in downsample_block.resnets:
                x = resnet(x)

            if i != (self.config.num_blocks - 1):
                # padding 0 in the downsampler conv so
                pad = (0, 1, 0, 1)
                x = F.pad(x, pad, mode="constant", value=0)
                for downsamplers in downsample_block.downsamplers:
                    x = downsamplers(x)
            out_states += (x,)

        x = self.mid_block.resnets[0](x)
        output_mid = (x,)
        for (
            attention,
            resnet,
        ) in zip(self.mid_block.attentions, self.mid_block.resnets[1:]):
            x = attention(x)
            output_mid += (x,)
            x = resnet(x)
            output_mid += (x,)

        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x, out_states, output_mid


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        block_out_channels = config.block_out_channels

        self.conv_in = nn.Conv2d(
            config.latent_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        num_layers = config.layers_per_block + 1
        for i in range(config.num_blocks):
            input_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            resnets = nn.ModuleList([])
            for j in range(num_layers):
                res_input_channel = input_channel if j == 0 else output_channel
                resnets.append(
                    ResnetBlock(
                        res_input_channel,
                        output_channel,
                        config.norm_num_groups,
                        config.norm_eps,
                    )
                )
            # Create a module dictionary for the entire block
            up_block = nn.ModuleDict(
                {
                    "resnets": resnets,
                }
            )
            if i < config.num_blocks - 1:  # No upsampler needed for the last block
                up_block["upsamplers"] = nn.ModuleList(
                    [UpsamplerBlock(output_channel, output_channel)]
                )

            self.up_blocks.append(up_block)

        # last layer is a resnet -> attention -> resnet
        self.mid_block = nn.ModuleDict(
            dict(
                resnets=nn.ModuleList(
                    [
                        ResnetBlock(
                            reversed_block_out_channels[0],
                            reversed_block_out_channels[0],
                            config.norm_num_groups,
                            config.norm_eps,
                        )
                        for _ in range(config.layers_per_block)
                    ]
                ),
                attentions=nn.ModuleList(
                    [
                        Attention(
                            reversed_block_out_channels[0],
                            config.attention_head_dim,
                            config.norm_num_groups,
                            config.norm_eps,
                        )
                    ]
                ),
            )
        )

        self.conv_norm_out = nn.GroupNorm(
            num_groups=config.norm_num_groups, num_channels=output_channel, eps=1e-6
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            output_channel, config.in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = self.conv_in(x)

        x = self.mid_block.resnets[0](x)
        output_mid = (x,)
        for (
            attention,
            resnet,
        ) in zip(self.mid_block.attentions, self.mid_block.resnets[1:]):
            x = attention(x)
            output_mid += (x,)
            x = resnet(x)
            output_mid += (x,)

        out_states = (x,)
        for i, up_block in enumerate(self.up_blocks):
            for resnet in up_block.resnets:
                x = resnet(x)

            if i != (self.config.num_blocks - 1):
                for upsamplers in up_block.upsamplers:
                    x = upsamplers(x)

            out_states += (x,)

        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x, out_states, output_mid


@dataclass
class VaeConfig:
    in_channels: int = 3
    latent_channels: int = 4
    block_out_channels: ClassVar[list[int]] = [128, 256, 512, 512]
    cross_attention_dim: int = 768
    num_blocks: int = 4
    attention_head_dim: int = 1
    layers_per_block: int = 2
    norm_num_groups: int = 32
    norm_eps: int = 1e-06


class Vae(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.post_quant_conv = nn.Conv2d(
            config.latent_channels, config.latent_channels, kernel_size=1, stride=1
        )
        self.quant_conv = nn.Conv2d(
            config.latent_channels * 2,
            config.latent_channels * 2,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x, generator=None):
        x, output_states, output_mid = self.encoder(x)
        x = self.quant_conv(x)
        mean, log_var = torch.split(x, self.config.latent_channels, dim=1)

        z_pre = self.reparameterize(mean, log_var, generator)
        z = self.post_quant_conv(z_pre)
        x_hat, output_states_dec, output_mid_dec = self.decoder(z)

        return (
            x_hat,
            mean,
            log_var,
            z_pre,
            output_states,
            output_mid,
            output_states_dec,
            output_mid_dec,
        )

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor, generator: torch.Generator = None
    ) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        if generator is not None:
            eps = torch.randn(std.size(), generator=generator, device=std.device)
        else:
            eps = torch.rand_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.post_quant_conv(z)
        x_hat, output_states_dec, output_mid_dec = self.decoder(z)
        return x_hat

    def encode(self, x, generator=None):
        x, output_states, output_mid = self.encoder(x)
        x = self.quant_conv(x)
        mean, log_var = torch.split(x, self.config.latent_channels, dim=1)

        z_pre = self.reparameterize(mean, log_var, generator)
        return z_pre

    @classmethod
    def from_pretrained(cls, config, model_path):
        """
        Loads pretrained weights from a .safetensors file, handling
        potential differences in layer naming conventions.
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
                    # Copy the tensor
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

        print("Weights loaded successfully.")
        return model
