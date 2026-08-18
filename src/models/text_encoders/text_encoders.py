import abc
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModel


def encode_tokens_batch(
    input_ids: torch.Tensor,
    text_encoder: torch.nn.Module,
    tokenizer: Any,
    max_length: int,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Encodes batch of CLIP token IDs, chunking long sequences if needed."""
    tokenizer_max_length = getattr(tokenizer, "model_max_length", 77)

    if input_ids.shape[-1] > tokenizer_max_length:
        processed_ids = []
        for iids_single in input_ids:
            chunks = []
            for i in range(
                1,
                max_length - tokenizer_max_length + 2,
                tokenizer_max_length - 2,
            ):
                ids_chunk = torch.cat(
                    [
                        iids_single[0].unsqueeze(0),
                        iids_single[i : i + tokenizer_max_length - 2],
                        iids_single[-1].unsqueeze(0),
                    ]
                )
                if (
                    ids_chunk[-2] != tokenizer.eos_token_id
                    and ids_chunk[-2] != tokenizer.pad_token_id
                ):
                    ids_chunk[-1] = tokenizer.eos_token_id
                if ids_chunk[1] == tokenizer.pad_token_id:
                    ids_chunk[1] = tokenizer.eos_token_id
                chunks.append(ids_chunk)
            processed_ids.append(torch.stack(chunks))

        processed_ids = torch.stack(processed_ids)
        batch_size, num_chunks, _ = processed_ids.shape
        input_ids_for_encoder = processed_ids.reshape(-1, tokenizer_max_length)
    else:
        input_ids_for_encoder = input_ids
        batch_size = input_ids.size(0)

    encoder_output = text_encoder(input_ids_for_encoder.to(device))
    if isinstance(encoder_output, (tuple, list)):
        text_embeddings = encoder_output[-1][-2]
    else:
        text_embeddings = encoder_output

    if hasattr(text_encoder, "text_model") and hasattr(
        text_encoder.text_model, "final_layer_norm"
    ):
        text_embeddings = text_encoder.text_model.final_layer_norm(text_embeddings)

    if input_ids.shape[-1] > tokenizer_max_length:
        text_embeddings = text_embeddings.reshape(
            batch_size, -1, text_embeddings.shape[-1]
        )
        states_list = [text_embeddings[:, 0].unsqueeze(1)]
        for i in range(1, max_length, tokenizer_max_length):
            states_list.append(text_embeddings[:, i : i + tokenizer_max_length - 2])
        states_list.append(text_embeddings[:, -1].unsqueeze(1))
        text_embeddings = torch.cat(states_list, dim=1)
        text_embeddings = text_embeddings[:, :max_length, :]

    return text_embeddings


# def _precompute_uncond(self):
#     # Helper to compute empty prompt embeddings for CFG
#     uncond_dict = {}
#     for tier in [77, 152, 227]:
#         tokens = self.tokenizer(
#             [""],
#             padding="max_length",
#             max_length=tier,
#             truncation=True,
#             return_tensors="pt",
#         ).input_ids
#
#         with torch.autocast(
#             device_type="cuda", dtype=self.autocast_dtype, enabled=True
#         ):
#             with torch.no_grad():
#                 te = (
#                     self.text_encoder.module
#                     if isinstance(self.text_encoder, DDP)
#                     else self.text_encoder
#                 )
#                 embeds = encode_tokens_batch(
#                     tokens.to(self.device), te, self.tokenizer, tier, self.device
#                 )
#                 uncond_dict[tier] = embeds.detach()
#     return uncond_dict
#


class BaseTextEncoder(nn.Module, abc.ABC):
    """Abstract interface for text encoder models."""

    @abc.abstractmethod
    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Outputs hidden states and boolean attention mask."""
        pass

    @property
    @abc.abstractmethod
    def embed_dim(self) -> int:
        pass


class HFTextEncoder(BaseTextEncoder):
    """Generic HuggingFace LLM / CausalLM Text Encoder Adapter."""

    def __init__(
        self,
        model_id: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: str = None,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self._embed_dim = self.model.config.hidden_size

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.model.parameters()).device
        tokens = tokens.to(device)
        if mask is not None:
            mask = mask.to(device)

        if drop_mask is not None and drop_mask.any():
            # WARNING: cfg must be empty prompt with its bos, pad only attention mask
            pad_id = getattr(self.model.config, "pad_token_id", 0) or 0
            tokens = tokens.clone()
            tokens[drop_mask] = pad_id
            if mask is not None:
                mask = mask.clone()
                mask[drop_mask] = False
                mask[drop_mask, 0] = True

        with torch.no_grad():
            outputs = self.model(
                input_ids=tokens,
                attention_mask=mask,
                return_dict=True,
            )
            embeddings = outputs.last_hidden_state

        if mask is None:
            mask = torch.ones(tokens.shape, dtype=torch.bool, device=tokens.device)
        return embeddings, mask


class CLIPTextEncoderWrapper(BaseTextEncoder):
    """CLIP text encoder adapter supporting SD1.5 legacy chunking."""

    def __init__(self, clip_model: nn.Module, tokenizer: Any):
        super().__init__()
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self._embed_dim = getattr(getattr(clip_model, "config", None), "n_embd", 768)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.clip_model.parameters()).device
        tokens = tokens.to(device)
        max_len = tokens.shape[-1]

        if drop_mask is not None and drop_mask.any():
            eos_id = getattr(self.tokenizer, "eos_token_id", 49407)
            tokens = tokens.clone()
            tokens[drop_mask] = eos_id
            if mask is not None:
                mask = mask.clone()
                mask[drop_mask] = False
                mask[drop_mask, :2] = True

        embeddings = encode_tokens_batch(
            tokens,
            self.clip_model,
            self.tokenizer,
            max_length=max_len,
            device=device,
        )

        if mask is None:
            mask = torch.ones(tokens.shape, dtype=torch.bool, device=tokens.device)
        return embeddings, mask
