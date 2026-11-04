import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
import json
import logging

import numpy as np
from PIL import Image
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


PERSON_COUNT_REGEX = re.compile(r"\b(\d+)\+?\s*(?:girl|boy)s?\b", re.IGNORECASE)
DEFAULT_IMG_SIZE = (448, 448)


def extract_target_count(prompt: str) -> Optional[int]:
    """
    Infers target character count from standard Danbooru tags.
    Returns None if the prompt contains no explicit count constraints.
    """
    tags = [t.strip().lower() for t in prompt.split(",") if t.strip()]

    if "no_humans" in tags:
        return 0

    explicit_counts = []
    for tag in tags:
        match = PERSON_COUNT_REGEX.match(tag)
        if match:
            explicit_counts.append(int(match.group(1)))

    if explicit_counts:
        return sum(explicit_counts)

    if "solo" in tags or "1girl" in tags or "1boy" in tags:
        return 1

    return None


class AestheticClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 4,
        hidden_dims: int = 512,
        dropout_rate: float = 0.1,
    ):
        """
        Multi-Layer Perceptron for 4-class aesthetic classification.
        """
        super(AestheticClassifier, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.fc1 = nn.Linear(feature_dim, hidden_dims)
        self.bn1 = nn.BatchNorm1d(hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dims // 2)
        self.fc3 = nn.Linear(hidden_dims // 2, num_classes)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.drop(self.relu(x))
        x = self.fc3(x)
        return x


class PairClassifier(nn.Module):
    def __init__(
        self, feature_dim: int, hidden_dims: int = 512, dropout_rate: float = 0.1
    ):
        """
        Simple MLP that concatenates two image embeddings and predicts
        which one is better.
        Class 0: First image is better.
        Class 1: Second image is better.
        """
        super(PairClassifier, self).__init__()
        self.feature_dim = feature_dim

        # The input will be the concatenation of two feature vectors
        input_dim = feature_dim * 2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims, hidden_dims // 2),
            nn.BatchNorm1d(hidden_dims // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims // 2, 2),
        )

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb1: Tensor of shape (Batch, feature_dim)
            emb2: Tensor of shape (Batch, feature_dim)
        Returns:
            Logits of shape (Batch, 2)
        """
        # Concatenate the two embeddings along the feature dimension
        x = torch.cat([emb1, emb2], dim=1)
        return self.mlp(x)


class AnimeMultiReward(nn.Module):
    """
    Unified multi-head anime reward model sharing a single SwinV2 backbone.
    Evaluates tag alignment, continuous 4-class aesthetics, and pairwise win
    rates with chunked VRAM execution and proper BGR channel handling.
    """

    def __init__(
        self,
        swin_dir: str,
        four_cls_path: str,
        pair_model_path: str = None,
        device: torch.device = "cuda",
        yolo_model_path: Optional[str] = None,
        feature_dim: int = 1024,
        eval_batch_size: int = 16,
        yolo_conf: float = 0.45,
    ):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.eval_batch_size = eval_batch_size
        self.yolo_conf = yolo_conf

        # 1. SwinV2 Backbone and Tag Head
        self.backbone, self.tag_to_idx, self.tagger_input_size = (
            self._load_swinv2_and_tags(swin_dir)
        )
        self.backbone.to(device).eval()

        # 2. 4-Class Aesthetic MLP Head
        four_cls_checkpoint = torch.load(four_cls_path, map_location=self.device)
        adjusted_state_dict = {}
        for key, value in four_cls_checkpoint["model_state_dict"].items():
            if key.startswith("_orig_mod."):
                new_key = key[len("_orig_mod.") :]
                adjusted_state_dict[new_key] = value
            else:
                adjusted_state_dict[key] = value

        self.four_cls = AestheticClassifier(
            feature_dim=four_cls_checkpoint["feature_dim"],
            num_classes=four_cls_checkpoint["num_classes"],
            hidden_dims=four_cls_checkpoint.get("hidden_dims", 512),
            dropout_rate=four_cls_checkpoint.get("dropout_rate", 0.1),
        )
        self.four_cls.load_state_dict(adjusted_state_dict)
        self.four_cls.to(device).eval()

        # 3. Pairwise Classifier Head
        self.pair_cls = PairClassifier(feature_dim=feature_dim).to(device)
        self.pair_cls.load_state_dict(torch.load(pair_model_path, map_location=device))
        self.pair_cls.eval()

        # 4. YOLO Face Detector
        if yolo_model_path and os.path.exists(yolo_model_path):
            self.yolo = YOLO(yolo_model_path)
        else:
            self.yolo = None

        self.requires_grad_(False)

    def _load_swinv2_and_tags(
        self, model_dir: str
    ) -> Tuple[nn.Module, Dict[str, int], Tuple[int, int]]:
        checkpoint_path = os.path.join(model_dir, "model.safetensors")
        config_path = os.path.join(model_dir, "config.json")
        tags_csv = os.path.join(model_dir, "selected_tags.csv")

        tag_to_idx = {}
        if os.path.exists(tags_csv):
            with open(tags_csv, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    tag_to_idx[row["name"].strip().lower()] = idx

        if not all(
            os.path.exists(p) for p in [model_dir, checkpoint_path, config_path]
        ):
            raise FileNotFoundError(
                f"Model checkpoint or config missing in: {model_dir}"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model_name = config.get("architecture", "swinv2_base_window8_256")
        num_classes = config.get("num_classes", 0)
        model_kwargs = config.get("model_args", {})
        input_size_cfg = config.get("pretrained_cfg", {}).get("input_size")

        if isinstance(input_size_cfg, (list, tuple)) and len(input_size_cfg) == 3:
            input_size_hw = (input_size_cfg[1], input_size_cfg[2])
        else:
            input_size_hw = DEFAULT_IMG_SIZE

        # Pass model_kwargs to correctly set img_size=448 and window_size=14
        model = timm.create_model(
            model_name,
            checkpoint_path=checkpoint_path,
            num_classes=len(tag_to_idx) if tag_to_idx else num_classes,
            **model_kwargs,
        )
        return model, tag_to_idx, input_size_hw

    @staticmethod
    def prepare_images(
        images: Union[torch.Tensor, List[Image.Image]],
        target_size: Tuple[int, int] = (448, 448),
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Preprocesses images for SwinV2 on GPU/CPU in batched tensor format.
        Preserves aspect ratio, pads with white (1.0), converts RGB to BGR,
        and applies (x - 0.5) / 0.5 normalization matching the original logic.
        """
        if isinstance(images, list):
            tensor_list = []
            for img in images:
                if isinstance(img, Image.Image):
                    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
                    arr = arr / 255.0
                    t = torch.from_numpy(arr).permute(2, 0, 1)
                else:
                    t = img
                tensor_list.append(t)
            images = torch.stack(tensor_list, dim=0)

        if images.ndim == 3:
            images = images.unsqueeze(0)

        if device is not None:
            images = images.to(device)

        images = images.to(dtype=torch.float32)
        if images.max() > 1.0:
            images = images / 255.0

        b, c, h, w = images.shape
        target_h, target_w = target_size

        # Compute aspect-preserving scale factor
        ratio = min(target_w / w, target_h / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)

        resized = F.interpolate(
            images,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )

        # Calculate symmetric padding for white background (value=1.0)
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        padded = F.pad(
            resized,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=1.0,
        )

        # Convert RGB to BGR as expected by the SwinV2 tagger backbone
        bgr = padded[:, [2, 1, 0], :, :]

        # Normalize with mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]
        normalized = (bgr - 0.5) / 0.5
        return normalized

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Helper routing to the static batch tensor preprocessor."""
        return self.prepare_images(
            images,
            target_size=self.tagger_input_size,
            device=self.device,
        )

    @torch.no_grad()
    def forward_features(
        self, images: Union[torch.Tensor, List[Image.Image]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts pooled features and tag logits in chunks of eval_batch_size.
        """
        b = images.shape[0] if isinstance(images, torch.Tensor) else len(images)
        pooled_list, logits_list = [], []

        for i in range(0, b, self.eval_batch_size):
            chunk = images[i : i + self.eval_batch_size]
            if isinstance(chunk, torch.Tensor):
                chunk = chunk.to(self.device, non_blocking=True)
            x = self.prepare_images(
                chunk,
                target_size=self.tagger_input_size,
                device=self.device,
            )
            feats = self.backbone.forward_features(x)
            pooled = self.backbone.head.global_pool(feats)
            logits = self.backbone.head.fc(pooled)
            pooled_list.append(pooled)
            logits_list.append(logits)

        return torch.cat(pooled_list, dim=0), torch.cat(logits_list, dim=0)

    @torch.no_grad()
    def compute_tag_reward(
        self,
        tag_logits: torch.Tensor,
        prompts: List[str],
        metadata: List[Dict],
    ) -> torch.Tensor:
        b = tag_logits.shape[0]
        probs = torch.sigmoid(tag_logits)
        scores = torch.zeros(b, device=self.device, dtype=torch.float32)

        for i, prompt in enumerate(prompts):
            tags = [t.strip().lower() for t in prompt.split(",") if t.strip()]
            tag_indices = [self.tag_to_idx[t] for t in tags if t in self.tag_to_idx]

            if not tag_indices:
                scores[i] = 1.0
                continue

            weights = None
            if metadata and i < len(metadata) and "tag_weights" in metadata[i]:
                w_dict = metadata[i]["tag_weights"]
                weights = torch.tensor(
                    [w_dict.get(t, 1.0) for t in tags if t in self.tag_to_idx],
                    device=self.device,
                    dtype=torch.float32,
                )

            target_probs = probs[i, tag_indices]
            if weights is not None and weights.sum() > 0:
                scores[i] = torch.sum(target_probs * weights) / weights.sum()
            else:
                scores[i] = torch.mean(target_probs)

        return scores

    @torch.no_grad()
    def compute_aesthetic_reward(self, features: torch.Tensor) -> torch.Tensor:
        b = features.shape[0]
        scores_list = []
        scale = torch.tensor(
            [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
            device=self.device,
            dtype=torch.float32,
        )

        for i in range(0, b, self.eval_batch_size):
            chunk = features[i : i + self.eval_batch_size]
            logits = self.four_cls(chunk)
            probs = F.softmax(logits, dim=-1)
            scores_list.append(torch.sum(probs * scale, dim=-1))

        return torch.cat(scores_list, dim=0)

    @torch.no_grad()
    def compute_pairwise_reward(
        self, features: torch.Tensor, prompts: List[str]
    ) -> torch.Tensor:
        b = features.shape[0]
        scores = torch.full((b,), 0.5, device=self.device, dtype=torch.float32)

        groups = defaultdict(list)
        for idx, p in enumerate(prompts):
            groups[p].append(idx)

        for _, indices in groups.items():
            k = len(indices)
            if k <= 1:
                continue

            group_idx = torch.tensor(indices, device=self.device)
            group_feats = features[group_idx]

            idx1 = torch.arange(k, device=self.device).repeat_interleave(k)
            idx2 = torch.arange(k, device=self.device).repeat(k)

            mask = idx1 != idx2
            p1_idx, p2_idx = idx1[mask], idx2[mask]

            emb1 = group_feats[p1_idx]
            emb2 = group_feats[p2_idx]

            logits_list = []
            for i in range(0, emb1.shape[0], self.eval_batch_size):
                e1 = emb1[i : i + self.eval_batch_size]
                e2 = emb2[i : i + self.eval_batch_size]
                logits_list.append(self.pair_cls(e1, e2))

            pair_logits = torch.cat(logits_list, dim=0)
            p_a_wins = F.softmax(pair_logits, dim=-1)[:, 0]

            pair_mat = torch.zeros((k, k), device=self.device)
            pair_mat[p1_idx, p2_idx] = p_a_wins

            sym_mat = 0.5 * (pair_mat + (1.0 - pair_mat.T))
            sym_mat.fill_diagonal_(0.0)

            win_rates = sym_mat.sum(dim=1) / (k - 1)
            scores[group_idx] = win_rates

        return scores

    @torch.no_grad()
    def compute_count_reward(
        self,
        images: torch.Tensor,
        prompts: List[str],
        metadata: List[Dict],
    ) -> torch.Tensor:
        """
        Calculates character count accuracy using YOLO anime face detection.
        Penalizes extra or missing characters against prompted targets.
        """
        b = images.shape[0]
        if self.yolo is None:
            return torch.ones(b, device=self.device, dtype=torch.float32)

        scores = torch.ones(b, device=self.device, dtype=torch.float32)

        for i in range(0, b, self.eval_batch_size):
            # Convert and transfer only the micro-chunk to GPU uint8
            raw_chunk = images[i : i + self.eval_batch_size]
            chunk_uint8 = (
                (raw_chunk.clamp(0, 1) * 255.0)
                .to(dtype=torch.uint8)
                .to(self.device, non_blocking=True)
            )

            results = self.yolo(
                chunk_uint8,
                conf=self.yolo_conf,
                device=self.device,
                verbose=False,
            )

            for local_idx, res in enumerate(results):
                global_idx = i + local_idx
                print(f"global_idx {global_idx}")
                prompt = prompts[global_idx]

                # fallback to prompt regex
                target = None
                if (
                    metadata
                    and global_idx < len(metadata)
                    and "target_count" in metadata[global_idx]
                ):
                    target = metadata[global_idx]["target_count"]
                else:
                    target = extract_target_count(prompt)

                if target is None:
                    scores[global_idx] = 1.0
                    continue

                detected = len(res.boxes)
                if target == 0:
                    scores[global_idx] = (
                        1.0 if detected == 0 else max(0.0, 1.0 - 0.5 * detected)
                    )
                else:
                    err = abs(detected - target)
                    scores[global_idx] = max(0.0, 1.0 - (err / float(target)))

        return scores


def multi_score(device: torch.device, reward_config: Dict):
    model = AnimeMultiReward(
        swin_dir=reward_config.get("swin_dir", "model/wd_swinv2"),
        four_cls_path=reward_config.get(
            "four_cls_path", "models/four_class_classifier_best.pth"
        ),
        pair_model_path=reward_config.get(
            "pair_model_path", "models/pair_classifier_best.pth"
        ),
        yolo_model_path=reward_config.get("yolo_model_path", "models/yolov8n-face.pt"),
        device=device,
        feature_dim=reward_config.get("feature_dim", 1024),
        eval_batch_size=reward_config.get("eval_batch_size", 16),
        yolo_conf=reward_config.get("yolo_conf", 0.45),
    )

    w_align = reward_config.get("w_align", 0.35)
    w_aes = reward_config.get("w_aes", 0.25)
    w_pair = reward_config.get("w_pair", 0.25)
    w_count = reward_config.get("w_count", 0.15)

    def reward_fn(
        images: torch.Tensor,
        prompts: List[str],
        prompt_metadata: List[Dict],
        only_strict: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        features, tag_logits = model.forward_features(images)

        r_tag = model.compute_tag_reward(tag_logits, prompts, prompt_metadata)
        r_aes = model.compute_aesthetic_reward(features)
        r_pair = model.compute_pairwise_reward(features, prompts)
        r_cnt = model.compute_count_reward(images, prompts, prompt_metadata)

        r_total = w_align * r_tag + w_aes * r_aes + w_pair * r_pair + w_count * r_cnt

        rewards = {
            "avg": r_total.cpu().numpy(),
            "tag_alignment": r_tag.cpu().numpy(),
            "aesthetic": r_aes.cpu().numpy(),
            "pairwise_tournament": r_pair.cpu().numpy(),
            "face_count": r_cnt.cpu().numpy(),
        }

        metadata = {
            "mean_tag_score": float(r_tag.mean().item()),
            "mean_aesthetic_score": float(r_aes.mean().item()),
            "mean_pairwise_winrate": float(r_pair.mean().item()),
            "mean_face_count_score": float(r_cnt.mean().item()),
        }

        return rewards, metadata

    return reward_fn
