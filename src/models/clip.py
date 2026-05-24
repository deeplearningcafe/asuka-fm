import safetensors
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import re


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sigmoid(1.702 * input)


class ClipEmbeddings(nn.Module):
    def __init__(self, n_embd, vocab_size, max_position_embs):
        super().__init__()
        self.max_position_embs = max_position_embs

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(max_position_embs, n_embd)

        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embs).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        input_ids,
    ):
        seq_length = input_ids.shape[1]
        if seq_length > self.max_position_embs:
            raise ValueError(
                f"Sequence length must be less than max_position_embeddings (got `sequence length`: "
                f"{seq_length} and max_position_embeddings: {self.max_position_embs}"
            )

        position_ids = self.position_ids[:, :seq_length]
        position_embs = self.position_embedding(position_ids)
        input_embs = self.token_embedding(input_ids)
        input_embs = input_embs + position_embs

        return input_embs


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()

        self.n_head = n_head

        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)

        self.out_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.activation_fn = QuickGELUActivation()
        self.fc2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, ln_eps):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(n_embd, eps=ln_eps)
        self.self_attn = CausalSelfAttention(n_embd, n_head)
        self.layer_norm2 = nn.LayerNorm(n_embd, eps=ln_eps)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))

        return x


@dataclass
class ClipConfig:
    block_size: int = 77
    vocab_size: int = 49408
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    ln_eps: float = 1e-05
    projection_dim: int = 768


class Clip(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.text_model = nn.ModuleDict(
            dict(
                embeddings=ClipEmbeddings(
                    config.n_embd, config.vocab_size, config.block_size
                ),
                encoder=nn.ModuleList(
                    [
                        TransformerBlock(config.n_embd, config.n_head, config.ln_eps)
                        for _ in range(self.config.n_layer)
                    ]
                ),
                final_layer_norm=nn.LayerNorm(config.n_embd, eps=config.ln_eps),
            )
        )

    def forward(self, input_ids):
        B, T = input_ids.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        )
        x = self.text_model.embeddings(input_ids)
        hidden_states = (x,)

        for block in self.text_model.encoder:
            x = block(x)
            hidden_states += (x,)
        x = self.text_model.final_layer_norm(x)

        return x, hidden_states

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

            k_custom = re.sub(r"\.encoder.layers\.", r".encoder.", k_custom)

            if k_custom in sd_custom:
                # Check shape compatibility
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

        print("Weights loaded successfully.")
        return model
