import abc
import random
import torch
from transformers import AutoTokenizer


class BaseTokenizer(abc.ABC):
    """Abstract base class defining the tokenizer interface."""

    @abc.abstractmethod
    def get_length(self, prompt: str) -> int:
        """Calculate length of a single prompt for sequence analysis."""
        pass

    @abc.abstractmethod
    def encode(
        self,
        prompt: str,
        max_len: int,
        cfg_dropout_prob: float = 0.0,
        tag_dropout_prob: float = 0.0,
        shuffle_tags: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes prompt into token tensor and attention mask."""
        pass

    @property
    @abc.abstractmethod
    def vocab(self) -> dict | None:
        """Returns vocabulary dictionary if available."""
        pass

    @abc.abstractmethod
    def decode(self, tokens):
        pass


class CommaSeparatedTokenizer(BaseTokenizer):
    """Rule-based tag tokenizer with shuffling and dropout."""

    def __init__(self, vocab: dict | None = None):
        self._vocab = vocab if vocab is not None else {"<pad>": 0, "<unk>": 1}
        self.pad_id = self._vocab.get("<pad>", 0)
        self.unk_id = self._vocab.get("<unk>", 1)

    def build_vocab(self, prompts: list[str]) -> None:
        self._vocab = {"<pad>": 0, "<unk>": 1}
        for prompt in prompts:
            tags = [t.strip() for t in prompt.split(",") if t.strip()]
            for tag in tags:
                if tag not in self._vocab:
                    self._vocab[tag] = len(self._vocab)
        self.pad_id = self._vocab.get("<pad>", 0)
        self.unk_id = self._vocab.get("<unk>", 1)

    @property
    def vocab(self) -> dict:
        return self._vocab

    def get_length(self, prompt: str) -> int:
        return len([t for t in prompt.split(",") if t.strip()])

    def encode(
        self,
        prompt: str,
        max_len: int,
        cfg_dropout_prob: float = 0.0,
        tag_dropout_prob: float = 0.0,
        shuffle_tags: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cfg_dropout_prob > 0.0 and random.random() < cfg_dropout_prob:
            tags = []
        else:
            tags = [t.strip() for t in prompt.split(",") if t.strip()]
            if len(tags) > 5:
                prefix = tags[:5]
                rest = tags[5:]
                if tag_dropout_prob > 0.0:
                    rest = [t for t in rest if random.random() >= tag_dropout_prob]
                if shuffle_tags:
                    random.shuffle(rest)
                tags = prefix + rest

        ids = [self._vocab.get(t, self.unk_id) for t in tags][:max_len]
        padded = ids + [self.pad_id] * (max_len - len(ids))
        token_tensor = torch.tensor(padded, dtype=torch.long)

        not_pad = token_tensor != self.pad_id
        shifted = torch.roll(not_pad, shifts=1, dims=0)
        shifted[0] = True
        attention_mask = not_pad | shifted
        return token_tensor, attention_mask


class HFLLMTokenizer(BaseTokenizer):
    """HuggingFace pretrained LLM tokenizer adapter."""

    def __init__(self, model_id: str, cache_dir: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"
        self.pad_id = self.tokenizer.pad_token_id

    @property
    def vocab(self) -> dict | None:
        return None

    def get_length(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt, add_special_tokens=True))

    def encode(
        self,
        prompt: str,
        max_len: int,
        cfg_dropout_prob: float = 0.0,
        tag_dropout_prob: float = 0.0,
        shuffle_tags: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cfg_dropout_prob > 0.0 and random.random() < cfg_dropout_prob:
            processed = ""
        else:
            # TODO: metadata ends with aesthetic or year
            tags = [t.strip() for t in prompt.split(",") if t.strip()]
            if len(tags) > 5:
                prefix = tags[:5]
                rest = tags[5:]
                if tag_dropout_prob > 0.0:
                    rest = [t for t in rest if random.random() >= tag_dropout_prob]
                if shuffle_tags:
                    random.shuffle(rest)
                tags = prefix + rest
            processed = ", ".join(tags)

        encoded = self.tokenizer(
            processed,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(
            0
        ).bool()

    def decode(self, tokens):
        prompts = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return prompts
