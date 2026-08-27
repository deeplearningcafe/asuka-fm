import math
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import IterableDataset

from src.data.dataset import (
    H5LatentDataset,
    LENGTH_TO_TIER_IDX,
    NUM_TIERS,
    TIER_LENGTHS,
)

AESTHETIC_TIER_MAP = {-1: 4, 0: 0, 1: 1, 2: 2, 3: 3}


class BaseBatchSampler(Sampler[List[int]], ABC):
    """
    Abstract base class for batch samplers with DDP rank partitioning
    and reproducible epoch-based seed management.
    """

    def __init__(
        self,
        dataset_len: int,
        dataset_ref: Optional[Dataset] = None,
        world_size: int = 1,
        rank: int = 0,
        seed: Optional[int] = None,
        drop_last: bool = False,
    ):
        self.world_size = world_size
        self.rank = rank
        self.drop_last = drop_last
        self.epoch = 0

        self.generator = torch.Generator()
        self.seed = seed if seed is not None else torch.seed()
        self.generator.manual_seed(self.seed + rank)

        if dataset_len == 0:
            raise ValueError("Dataset cannot be empty.")

        self.num_samples_total = dataset_len
        if world_size > 1:
            if dataset_ref is not None:
                sampler = DistributedSampler(
                    dataset_ref,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    seed=self.seed,
                )
                self.indices = list(sampler)
            else:
                indices_all = list(range(dataset_len))
                self.indices = indices_all[rank::world_size]
            self.num_samples = len(self.indices)
        else:
            self.indices = list(range(dataset_len))
            self.num_samples = len(self.indices)

    def set_epoch(self, epoch: int):
        """Updates epoch state to vary pseudorandom shuffling per epoch."""
        self.epoch = epoch
        self.generator.manual_seed(self.seed + self.rank + epoch)

    @abstractmethod
    def __iter__(self) -> Iterator[List[int]]:
        """Yields batches of integer indices."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Returns the estimated number of batches per epoch."""
        pass


class StreamingTokenTierBatchSampler(IterableDataset):
    """
    Streaming tier batch sampler wrapping an IterableDataset. Buffers
    samples from streaming shards, groups strictly by token sequence
    length tier to eliminate padding waste, applies dynamic batch sizing,
    and executes aesthetic curriculum sampling.
    """

    AESTHETIC_TIER_MAP = AESTHETIC_TIER_MAP

    def __init__(
        self,
        dataset: IterableDataset,
        base_batch_size: int,
        tier_lengths: Optional[List[int]] = None,
        base_sequence_length: Optional[int] = None,
        length_penalty_power: float = 0.0,
        drop_last: bool = False,
        seed: Optional[int] = None,
        world_size: int = 1,
        rank: int = 0,
        aesthetic_curriculum: bool = True,
        min_batch_size: Optional[int] = None,
        buffer_size: int = 10000,
        refill_chunk_size: Optional[int] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.base_batch_size = base_batch_size
        self.tier_lengths = sorted(tier_lengths or TIER_LENGTHS)
        self.num_tiers = len(self.tier_lengths)
        self.tier_to_idx = {length: i for i, length in enumerate(self.tier_lengths)}
        self.base_sequence_length = base_sequence_length or self.tier_lengths[0]
        self.length_penalty_power = length_penalty_power
        self.drop_last = drop_last
        self.seed = seed if seed is not None else torch.seed()
        self.world_size = world_size
        self.rank = rank
        self.aesthetic_curriculum = aesthetic_curriculum
        self.min_batch_size = (
            min_batch_size
            if min_batch_size is not None
            else max(1, base_batch_size // 3)
        )
        self.refill_chunk_size = (
            refill_chunk_size
            if refill_chunk_size is not None
            else max(64, base_batch_size * 2)
        )
        self.buffer_size = buffer_size
        self.num_aesthetic_tiers = 5
        self.epoch = 0
        self.samples_yielded = 0
        self.total_samples = int(getattr(dataset, "total_samples", 0))

        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed + self.rank)

    def set_epoch(self, epoch: int):
        """Updates epoch seed and propagates to underlying dataset."""
        self.epoch = epoch
        self.samples_yielded = 0
        self.generator.manual_seed(self.seed + self.rank + epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        elif hasattr(getattr(self.dataset, "hf_dataset", None), "set_epoch"):
            self.dataset.hf_dataset.set_epoch(epoch)

    def _determine_tier_idx(self, seq_len: int) -> int:
        """Maps token sequence length to closest matching tier index."""
        if seq_len in self.tier_to_idx:
            return self.tier_to_idx[seq_len]
        for idx, max_len in enumerate(self.tier_lengths):
            if seq_len <= max_len:
                return idx
        return self.num_tiers - 1

    def _calculate_batch_size(self, tier_idx: int) -> int:
        """
        Calculates dynamic batch size based on sequence length.
        Longer prompt sequences require a smaller batch size to avoid OOM.
        """
        if self.length_penalty_power <= 0.0:
            return self.base_batch_size
        tier_length = self.tier_lengths[tier_idx]
        scale = (
            self.base_sequence_length / max(tier_length, 1)
        ) ** self.length_penalty_power
        return max(1, int(round(self.base_batch_size * min(scale, 1.0))))

    def _push_sample(
        self,
        sample: Any,
        bins: List[List[List[Any]]],
        tier_counts: List[int],
        aes_counts: List[List[int]],
    ) -> None:
        """Parses and slots a raw sample into its tier and aesthetic bin."""
        if isinstance(sample, dict):
            tier_val = int(sample.get("tier", self.tier_lengths[-1]))
            aes_raw = int(sample.get("aesthetic_tier", -1))
        elif isinstance(sample, (tuple, list)):
            tokens = sample[1]
            aes_val = sample[5]
            tier_val = (
                tokens.shape[-1]
                if hasattr(tokens, "shape")
                else len(tokens)
            )
            aes_raw = (
                int(aes_val.item())
                if hasattr(aes_val, "item")
                else int(aes_val)
            )
        else:
            tier_val = self.tier_lengths[-1]
            aes_raw = -1

        tier_idx = self._determine_tier_idx(tier_val)
        aes_idx = self.AESTHETIC_TIER_MAP.get(aes_raw, 4)

        bins[tier_idx][aes_idx].append(sample)
        aes_counts[tier_idx][aes_idx] += 1
        tier_counts[tier_idx] += 1

    def _refill_buffer(
        self,
        stream_iter: Iterator[Any],
        bins: List[List[List[Any]]],
        tier_counts: List[int],
        aes_counts: List[List[int]],
        max_samples_to_fetch: int,
    ) -> bool:
        """
        Fetches up to max_samples_to_fetch into bins.
        Returns True if the underlying stream is exhausted.
        """
        for _ in range(max_samples_to_fetch):
            try:
                sample = next(stream_iter)
            except StopIteration:
                return True
            self._push_sample(sample, bins, tier_counts, aes_counts)
        return False

    def _collate_batch(
        self, batch_samples: List[Tuple[torch.Tensor, ...]]
    ) -> Tuple[torch.Tensor, ...]:
        """Stacks individual sample tensors into aligned batch tensors."""
        processed_batchs = []
        for sample in batch_samples:
            processed_batchs.append(self.dataset._process_sample(sample))
        images = torch.stack([s[0] for s in processed_batchs], dim=0)
        tokens = torch.stack([s[1] for s in processed_batchs], dim=0)
        mask = torch.stack([s[2] for s in processed_batchs], dim=0)
        pos_map = torch.stack([s[3] for s in processed_batchs], dim=0)
        tag_weights = torch.stack(
            [s[4] if s[4].ndim > 0 else s[4].unsqueeze(0) for s in processed_batchs],
            dim=0,
        ).squeeze(-1)
        aes_tier = torch.stack(
            [s[5] if s[5].ndim > 0 else s[5].unsqueeze(0) for s in processed_batchs],
            dim=0,
        ).squeeze(-1)

        return images, tokens, mask, pos_map, tag_weights, aes_tier

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        try:
            import pyarrow as pa

            pa_pool = pa.default_memory_pool()
        except ImportError:
            pa_pool = None

        stream_iter = (
            self.dataset.iter_raw()
            if hasattr(self.dataset, "iter_raw")
            else iter(self.dataset)
        )

        # Persistent bins: [tier_idx][aesthetic_tier_idx]
        bins: List[List[List[Any]]] = [
            [[] for _ in range(self.num_aesthetic_tiers)]
            for _ in range(self.num_tiers)
        ]
        tier_counts = [0] * self.num_tiers
        aes_counts = [
            [0] * self.num_aesthetic_tiers for _ in range(self.num_tiers)
        ]

        # Initial large prefill to stock the high-RAM shock absorber
        stream_exhausted = self._refill_buffer(
            stream_iter=stream_iter,
            bins=bins,
            tier_counts=tier_counts,
            aes_counts=aes_counts,
            max_samples_to_fetch=self.buffer_size,
        )

        while True:
            ready_tiers = [
                i
                for i, count in enumerate(tier_counts)
                if count >= self._calculate_batch_size(i)
            ]

            # Low-watermark recovery: fetch incremental chunks if not ready
            while not ready_tiers and not stream_exhausted:
                stream_exhausted = self._refill_buffer(
                    stream_iter=stream_iter,
                    bins=bins,
                    tier_counts=tier_counts,
                    aes_counts=aes_counts,
                    max_samples_to_fetch=self.refill_chunk_size,
                )
                ready_tiers = [
                    i
                    for i, count in enumerate(tier_counts)
                    if count >= self._calculate_batch_size(i)
                ]
                if sum(tier_counts) >= self.buffer_size:
                    break

            if not ready_tiers:
                if stream_exhausted:
                    break
                # Drain remaining samples if buffer full without full tiers
                non_empty = [i for i, c in enumerate(tier_counts) if c > 0]
                if not non_empty:
                    break
                chosen_tier = max(non_empty, key=lambda i: tier_counts[i])
                target_bs = min(
                    tier_counts[chosen_tier],
                    self._calculate_batch_size(chosen_tier),
                )
            else:
                ready_weights = torch.tensor(
                    [tier_counts[i] for i in ready_tiers],
                    dtype=torch.float32,
                )
                slot = torch.multinomial(
                    ready_weights, 1, generator=self.generator
                ).item()
                chosen_tier = ready_tiers[slot]
                target_bs = self._calculate_batch_size(chosen_tier)

            # batch with exact target_bs using aesthetic curriculum
            batch_samples = []
            while len(batch_samples) < target_bs:
                avail_aes = [
                    a
                    for a in range(self.num_aesthetic_tiers)
                    if aes_counts[chosen_tier][a] > 0
                ]
                if not avail_aes:
                    break

                if self.aesthetic_curriculum:
                    prog = min(
                        1.0,
                        self.samples_yielded / max(1, self.total_samples),
                    )
                    p_simple = (1.0 - prog) * 0.4 + prog * 0.25
                    p_complex = (1.0 - prog) * 0.1 + prog * 0.25
                    weights = [
                        p_simple if a in (1, 2) else p_complex for a in avail_aes
                    ]
                else:
                    weights = [float(aes_counts[chosen_tier][a]) for a in avail_aes]

                aes_tensor = torch.tensor(weights, dtype=torch.float32)
                if aes_tensor.sum() <= 0 or not torch.all(torch.isfinite(aes_tensor)):
                    aes_tensor = torch.ones(len(avail_aes), dtype=torch.float32)

                aes_slot = torch.multinomial(
                    aes_tensor, 1, generator=self.generator
                ).item()
                chosen_aes = avail_aes[aes_slot]

                num_needed = target_bs - len(batch_samples)
                num_avail = aes_counts[chosen_tier][chosen_aes]
                take = min(num_needed, num_avail)

                src = bins[chosen_tier][chosen_aes]
                for _ in range(take):
                    batch_samples.append(src.pop())

                aes_counts[chosen_tier][chosen_aes] -= take

            # ensure full target_bs
            if len(batch_samples) < target_bs:
                for a in range(self.num_aesthetic_tiers):
                    src = bins[chosen_tier][a]
                    while src and len(batch_samples) < target_bs:
                        batch_samples.append(src.pop())
                        aes_counts[chosen_tier][a] -= 1

            tier_counts[chosen_tier] -= len(batch_samples)
            self.samples_yielded += len(batch_samples)

            # Continuous steady-state incremental refill (top-off)
            current_buffered = sum(tier_counts)
            if not stream_exhausted and current_buffered < self.buffer_size:
                needed = self.buffer_size - current_buffered
                fetch_count = min(self.refill_chunk_size, needed)
                stream_exhausted = self._refill_buffer(
                    stream_iter=stream_iter,
                    bins=bins,
                    tier_counts=tier_counts,
                    aes_counts=aes_counts,
                    max_samples_to_fetch=fetch_count,
                )
                if pa_pool is not None and self.samples_yielded % 200 == 0:
                    pa_pool.release_unused()

            yield self._collate_batch(batch_samples)

    def __len__(self) -> int:
        num_samples = getattr(
            self.dataset,
            "num_samples",
            getattr(self.dataset, "total_samples", 0),
        )
        if num_samples == 0:
            return 0
        return (num_samples + self.base_batch_size - 1) // max(1, self.base_batch_size)


class BucketBatchSampler(BaseBatchSampler):
    """
    A batch sampler that groups samples by aspect ratio bucket, prompt
    length tier, and aesthetic tier. Supports dynamic batch sizing and
    distributed training.
    """

    def __init__(
        self,
        dataset: H5LatentDataset,
        base_batch_size: int,
        base_resolution_area: int = 64 * 64,
        base_sequence_length: int = 77,
        length_penalty_power: float = 2.0,
        drop_last: bool = False,
        seed: Optional[int] = None,
        max_batch_size_ratio: float = 4.0,
        world_size: int = 1,
        rank: int = 0,
        initial_epoch_focus_low_res: int = 0,
        low_res_focus_factor: float = 1.0,
        low_res_area_percentile: float = 0.33,
    ):
        """
        Initializes the BucketBatchSampler.

        Args:
            dataset: Dataset containing latent representations and metadata.
            base_batch_size: Target batch size for base resolution.
            base_resolution_area: Pixel area mapping to base batch size.
            base_sequence_length: Reference sequence length for scaling.
            length_penalty_power: Power factor for sequence length scaling.
            drop_last: Prunes tiers that cannot form a complete batch.
            seed: Random seed for reproducibility.
            max_batch_size_ratio: Upper bound factor for dynamic batch size.
            world_size: Number of distributed processes.
            rank: Rank of the current process.
            initial_epoch_focus_low_res: Epochs to prioritize low-res.
            low_res_focus_factor: Probability multiplier for low-res.
            low_res_area_percentile: Cutoff percentile for low-res.
        """
        super().__init__(
            dataset_len=len(dataset),
            dataset_ref=dataset,
            world_size=world_size,
            rank=rank,
            seed=seed,
            drop_last=drop_last,
        )

        self.dataset = dataset
        self.base_batch_size = base_batch_size
        self.base_resolution_area = base_resolution_area
        if base_sequence_length not in LENGTH_TO_TIER_IDX:
            raise ValueError(
                f"base_sequence_length ({base_sequence_length}) "
                f"must be one of {TIER_LENGTHS}"
            )
        self.base_sequence_length = base_sequence_length
        self.length_penalty_power = length_penalty_power
        self.max_batch_size_ratio = max_batch_size_ratio

        self.buckets_idx_list = [bucket["bucket_idx"] for bucket in dataset.bucket_info]
        self.num_buckets = len(dataset.bucket_info)
        if self.num_buckets == 0:
            raise ValueError("Dataset metadata lacks bucket information.")

        self.num_tiers = NUM_TIERS
        self.tier_lengths = TIER_LENGTHS
        self.num_aesthetic_tiers = 5

        self.bucket_id_to_internal_idx: Dict[int, int] = {}
        self.bucket_latent_areas = []

        # Map global dataset index to bucket index
        self.index_to_bucket_map: Dict[int, int] = {}
        for i, b_info in enumerate(dataset.bucket_info):
            self.bucket_id_to_internal_idx[b_info["bucket_idx"]] = i
            lat_res = b_info.get("latents_resolution")
            if lat_res is None or len(lat_res) != 2:
                raise ValueError(f"Invalid 'latents_resolution' for bucket {i}.")
            self.bucket_latent_areas.append(lat_res[0] * lat_res[1])

        self.initial_epoch_focus_low_res = initial_epoch_focus_low_res
        self.low_res_focus_factor = low_res_focus_factor
        self.low_res_area_percentile = low_res_area_percentile
        self.low_res_area_threshold = None

        if self.initial_epoch_focus_low_res > 0:
            if not (0 < self.low_res_area_percentile < 1):
                warnings.warn(
                    f"[Rank {rank}] low_res_area_percentile "
                    f"({self.low_res_area_percentile}) is not between 0 and 1. "
                    f"Disabling low-res focus."
                )
                self.initial_epoch_focus_low_res = 0
            elif self.low_res_focus_factor <= 0:
                warnings.warn(
                    f"[Rank {rank}] low_res_focus_factor "
                    f"({self.low_res_focus_factor}) is not positive. "
                    f"Disabling low-res focus."
                )
                self.initial_epoch_focus_low_res = 0
            else:
                # Determine threshold from all unique bucket areas in metadata
                all_unique_sorted_areas = sorted(list(set(self.bucket_latent_areas)))
                if all_unique_sorted_areas:
                    print(f"Unique areas {all_unique_sorted_areas}")
                    idx = int(
                        len(all_unique_sorted_areas) * self.low_res_area_percentile
                    )
                    idx = min(max(0, idx), len(all_unique_sorted_areas) - 1)
                    self.low_res_area_threshold = all_unique_sorted_areas[idx]

                    print(
                        f"[Rank {rank}] Initial epoch focus active for "
                        f"{self.initial_epoch_focus_low_res} epochs."
                    )
                    print(f"    Focus factor: {self.low_res_focus_factor}")
                    print(
                        f"    Targeting buckets with area <= "
                        f"{self.low_res_area_threshold} (based on "
                        f"{self.low_res_area_percentile * 100:.0f}th "
                        f"percentile of all bucket areas)."
                    )
                else:
                    warnings.warn(
                        f"[Rank {rank}] No bucket areas found to determine "
                        f"low_res_area_threshold. Disabling low-res focus."
                    )
                    self.initial_epoch_focus_low_res = 0

        # Shape: [bucket_idx][tier_idx][aesthetic_tier_idx] -> list_of_indices
        self.indices_per_bucket_tier_aes: List[List[List[List[int]]]] = [
            [
                [[] for _ in range(self.num_aesthetic_tiers)]
                for _ in range(self.num_tiers)
            ]
            for _ in range(self.num_buckets)
        ]
        # Shape: [bucket_idx][tier_idx][aesthetic_tier_idx] -> count
        self.samples_per_bucket_tier_aes: List[List[List[int]]] = [
            [[0] * self.num_aesthetic_tiers for _ in range(self.num_tiers)]
            for _ in range(self.num_buckets)
        ]
        self.indices_per_bucket_tier: List[List[List[int]]] = [
            [[] for _ in range(self.num_tiers)] for _ in range(self.num_buckets)
        ]

        valid_indices_count = 0
        skipped_invalid_tier = 0
        skipped_invalid_bucket = 0
        skipped_invalid_aes = 0
        for global_idx in self.indices:
            sample_meta = dataset.sample_mapping[global_idx]
            metadata_bucket_idx = sample_meta.get("bucket_idx")
            tier_len = sample_meta.get("tier")
            aes_tier_from_meta = sample_meta.get("aesthetic_tier", -1)

            bucket_idx = self.bucket_id_to_internal_idx.get(metadata_bucket_idx)

            if bucket_idx is None or bucket_idx > self.num_buckets:
                skipped_invalid_bucket += 1
                print(f"Invalid bucket idx {bucket_idx}")
                continue

            tier_idx = LENGTH_TO_TIER_IDX.get(tier_len)
            if tier_idx is None:
                skipped_invalid_tier += 1
                continue

            aes_tier_idx = AESTHETIC_TIER_MAP.get(aes_tier_from_meta)
            if aes_tier_idx is None:
                skipped_invalid_aes += 1
                continue

            self.indices_per_bucket_tier_aes[bucket_idx][tier_idx][aes_tier_idx].append(
                global_idx
            )
            self.samples_per_bucket_tier_aes[bucket_idx][tier_idx][aes_tier_idx] += 1
            self.indices_per_bucket_tier[bucket_idx][tier_idx].append(global_idx)
            valid_indices_count += 1

        self.samples_per_bucket_tier: List[List[int]] = [
            [sum(aes_counts) for aes_counts in tier_counts]
            for tier_counts in self.samples_per_bucket_tier_aes
        ]
        self.samples_per_bucket: List[int] = [
            sum(tier_counts) for tier_counts in self.samples_per_bucket_tier
        ]

        if skipped_invalid_bucket > 0:
            warnings.warn(
                f"[Rank {rank}] Skipped {skipped_invalid_bucket} samples "
                f"due to invalid bucket index."
            )
        if skipped_invalid_tier > 0:
            warnings.warn(
                f"[Rank {rank}] Skipped {skipped_invalid_tier} samples "
                f"due to unrecognized tier length."
            )
        if skipped_invalid_aes > 0:
            warnings.warn(
                f"[Rank {rank}] Skipped {skipped_invalid_aes} "
                f"samples due to invalid 'aesthetic_tier'."
            )

        if valid_indices_count == 0:
            raise ValueError(
                f"[Rank {rank}] No valid samples assigned to buckets/tiers."
            )

        self.active_bucket_indices_init = [
            i for i, count in enumerate(self.samples_per_bucket) if count > 0
        ]
        if not self.active_bucket_indices_init:
            raise ValueError(f"[Rank {rank}] No samples found in any bucket.")

        print(
            f"[Rank {rank}] BucketBatchSampler initialized: "
            f"{len(self.active_bucket_indices_init)} active buckets, "
            f"{self.num_tiers} tiers, {self.num_aesthetic_tiers} "
            f"aesthetic tiers for {self.num_samples} samples."
        )

    def _calculate_batch_size(self, bucket_idx: int, tier_idx: int) -> int:
        """Calculates dynamic batch size based on area and sequence length."""
        latent_area = self.bucket_latent_areas[bucket_idx]
        tier_length = self.tier_lengths[tier_idx]

        if latent_area <= 0 or tier_length <= 0:
            return 1

        area_ratio = self.base_resolution_area / latent_area
        clamped_area_ratio = min(area_ratio, self.max_batch_size_ratio)

        # 2. Length scaling penalty
        length_penalty = (
            self.base_sequence_length / max(tier_length, 1)
        ) ** self.length_penalty_power
        length_penalty = min(length_penalty, 1.0)

        dynamic_bs = max(
            1, int(self.base_batch_size * clamped_area_ratio * length_penalty)
        )

        return dynamic_bs

    def __iter__(self) -> Iterator[List[int]]:
        """
        Generates batches by prioritizing or scheduling combinations of
        buckets, sequence length tiers, and aesthetic tiers.
        """
        # internal bucket indices
        current_active_buckets = list(self.active_bucket_indices_init)
        if not current_active_buckets:
            return iter([])

        remaining_indices = [
            [
                [list(aes_indices) for aes_indices in tier_indices]
                for tier_indices in bucket_indices
            ]
            for bucket_indices in self.indices_per_bucket_tier_aes
        ]
        remaining_counts_aes = [
            [list(aes_counts) for aes_counts in tier_counts]
            for tier_counts in self.samples_per_bucket_tier_aes
        ]
        remaining_indices_per_bucket_tier = [
            [list(tier_indices) for tier_indices in bucket_indices]
            for bucket_indices in self.indices_per_bucket_tier
        ]
        remaining_counts_per_bucket_tier = [
            list(tier_counts) for tier_counts in self.samples_per_bucket_tier
        ]

        # Define the minimum number of samples required to form a batch.
        min_batch_size = self.base_batch_size // 3

        if min_batch_size > 0:
            # Prune tiers that are too small to ever form a valid batch.
            for b_idx in range(self.num_buckets):
                for t_idx in range(self.num_tiers):
                    if (
                        0
                        < remaining_counts_per_bucket_tier[b_idx][t_idx]
                        < min_batch_size
                    ):
                        # Drop these samples for this epoch.
                        remaining_indices_per_bucket_tier[b_idx][t_idx].clear()
                        remaining_counts_per_bucket_tier[b_idx][t_idx] = 0

        remaining_samples_per_bucket = [
            sum(tier_counts) for tier_counts in remaining_counts_per_bucket_tier
        ]
        current_active_buckets = [
            i for i, count in enumerate(remaining_samples_per_bucket) if count > 0
        ]
        num_remaining_total = sum(remaining_samples_per_bucket)

        # If no samples are left after pruning, end the epoch immediately.
        if not current_active_buckets:
            return iter([])

        # Shuffle indices within each (bucket, tier, aes_tier) combination.
        for b_idx in current_active_buckets:
            for t_idx in range(self.num_tiers):
                for a_idx in range(self.num_aesthetic_tiers):
                    indices_list = remaining_indices[b_idx][t_idx][a_idx]
                    if indices_list:
                        perm = torch.randperm(
                            len(indices_list), generator=self.generator
                        )
                        indices_list[:] = [indices_list[i] for i in perm.tolist()]

        num_remaining_total = sum(remaining_samples_per_bucket)
        # Store initial count to calculate epoch progress.
        initial_num_samples = num_remaining_total

        # Batch Generation Loop
        while num_remaining_total > 0:
            if not current_active_buckets:
                break

            # Calculate initial weights based on remaining samples.
            bucket_raw_weights = []
            current_total_raw_weight = 0.0

            apply_low_res_focus = (
                self.epoch < self.initial_epoch_focus_low_res
                and self.low_res_area_threshold is not None
                and self.low_res_focus_factor > 1.0
            )

            for original_bucket_idx in current_active_buckets:
                weight = float(remaining_samples_per_bucket[original_bucket_idx])

                if apply_low_res_focus:
                    if (
                        self.bucket_latent_areas[original_bucket_idx]
                        <= self.low_res_area_threshold
                    ):
                        weight *= self.low_res_focus_factor

                bucket_raw_weights.append(weight)
                current_total_raw_weight += weight

            # Normalize the raw weights to get probabilities
            if current_total_raw_weight > 1e-9:
                bucket_probs = [
                    w / current_total_raw_weight for w in bucket_raw_weights
                ]
            else:
                if current_active_buckets:
                    num_active = len(current_active_buckets)
                    bucket_probs = [1.0 / num_active] * num_active
                    if self.rank == 0 and num_remaining_total > 0:
                        warnings.warn(
                            f"[Epoch {self.epoch}] Bucket weights sum to near zero, "
                            f"falling back to uniform bucket sampling. "
                            f"Remaining: {num_remaining_total}"
                        )
                else:
                    break

            bucket_weights_tensor = torch.tensor(bucket_probs, dtype=torch.float32)

            if (
                not torch.all(torch.isfinite(bucket_weights_tensor))
                or bucket_weights_tensor.sum() < 1e-6
            ):
                if self.rank == 0:
                    warnings.warn(
                        f"[Epoch {self.epoch}, Rank {self.rank}] Invalid bucket "
                        f"weights ({bucket_weights_tensor}), trying uniform."
                    )
                num_active = len(current_active_buckets)
                if num_active == 0:
                    break
                bucket_weights_tensor = (
                    torch.ones(num_active, dtype=torch.float32) / num_active
                )
                if not torch.all(torch.isfinite(bucket_weights_tensor)):
                    break

            chosen_active_bucket_idx_in_list = torch.multinomial(
                bucket_weights_tensor, 1, replacement=True, generator=self.generator
            ).item()
            chosen_original_bucket_idx = current_active_buckets[
                chosen_active_bucket_idx_in_list
            ]

            # Find tiers in the chosen bucket that still have samples
            active_tier_indices_in_bucket = [
                t_idx
                for t_idx in range(self.num_tiers)
                if remaining_counts_per_bucket_tier[chosen_original_bucket_idx][t_idx]
                > 0
            ]

            if not active_tier_indices_in_bucket:
                if remaining_samples_per_bucket[chosen_original_bucket_idx] == 0:
                    current_active_buckets.pop(chosen_active_bucket_idx_in_list)
                continue

            total_weight_tiers = float(
                sum(
                    remaining_counts_per_bucket_tier[chosen_original_bucket_idx][t_idx]
                    for t_idx in active_tier_indices_in_bucket
                )
            )

            if total_weight_tiers <= 0:
                continue

            tier_weights = torch.tensor(
                [
                    remaining_counts_per_bucket_tier[chosen_original_bucket_idx][t_idx]
                    / total_weight_tiers
                    for t_idx in active_tier_indices_in_bucket
                ],
                dtype=torch.float32,
            )

            if not torch.all(torch.isfinite(tier_weights)) or tier_weights.sum() < 1e-6:
                warnings.warn(
                    f"[Rank {self.rank}] Invalid tier weights in bucket "
                    f"{chosen_original_bucket_idx}, falling back to uniform."
                )
                tier_weights = torch.ones_like(tier_weights) / len(tier_weights)
                if not torch.all(torch.isfinite(tier_weights)):
                    continue

            chosen_active_tier_idx_in_list = torch.multinomial(
                tier_weights, 1, replacement=True, generator=self.generator
            ).item()
            chosen_tier_idx = active_tier_indices_in_bucket[
                chosen_active_tier_idx_in_list
            ]

            # Calculate Batch Size & Fetch Indices
            target_batch_size = self._calculate_batch_size(
                chosen_original_bucket_idx, chosen_tier_idx
            )

            remaining_in_chosen_tier = remaining_counts_per_bucket_tier[
                chosen_original_bucket_idx
            ][chosen_tier_idx]
            actual_batch_size = min(target_batch_size, remaining_in_chosen_tier)

            if actual_batch_size <= 0:
                warnings.warn(
                    f"[Rank {self.rank}] Tried to sample from empty tier "
                    f"({chosen_original_bucket_idx}, {chosen_tier_idx})."
                )
                if remaining_in_chosen_tier == 0:
                    remaining_samples_per_bucket[chosen_original_bucket_idx] = sum(
                        remaining_counts_per_bucket_tier[chosen_original_bucket_idx]
                    )
                continue

            indices_list_to_pop_from = remaining_indices_per_bucket_tier[
                chosen_original_bucket_idx
            ][chosen_tier_idx]

            batch_indices = []
            while len(batch_indices) < actual_batch_size:
                available_aes = [
                    a_idx
                    for a_idx in range(self.num_aesthetic_tiers)
                    if remaining_counts_aes[chosen_original_bucket_idx][
                        chosen_tier_idx
                    ][a_idx]
                    > 0
                ]
                if not available_aes:
                    break

                progress = 1.0 - (num_remaining_total / initial_num_samples)
                # At progress=0, simple tiers (1,2) have 4x prob of complex (0,3)
                # At progress=1, all tiers have equal probability.
                p_simple = (1 - progress) * 0.4 + progress * 0.25
                p_complex = (1 - progress) * 0.1 + progress * 0.25

                aes_weights_list = []
                for a_idx in available_aes:
                    is_simple = a_idx in [1, 2]
                    aes_weights_list.append(p_simple if is_simple else p_complex)

                total_aes_weight = sum(aes_weights_list)
                aes_probs = [w / total_aes_weight for w in aes_weights_list]
                aes_weights = torch.tensor(aes_probs, dtype=torch.float32)

                chosen_active_aes_idx = torch.multinomial(
                    aes_weights, 1, replacement=True, generator=self.generator
                ).item()
                chosen_a_idx = available_aes[chosen_active_aes_idx]

                num_remaining_in_group = remaining_counts_aes[
                    chosen_original_bucket_idx
                ][chosen_tier_idx][chosen_a_idx]
                num_needed_for_batch = actual_batch_size - len(batch_indices)
                num_to_take = min(num_remaining_in_group, num_needed_for_batch)

                if num_to_take <= 0:
                    continue

                indices_list = remaining_indices[chosen_original_bucket_idx][
                    chosen_tier_idx
                ][chosen_a_idx]
                for _ in range(num_to_take):
                    batch_indices.append(indices_list.pop())

                remaining_counts_aes[chosen_original_bucket_idx][chosen_tier_idx][
                    chosen_a_idx
                ] -= num_to_take

            # Update State & Handle Remainders
            remaining_counts_per_bucket_tier[chosen_original_bucket_idx][
                chosen_tier_idx
            ] -= actual_batch_size
            remaining_samples_per_bucket[chosen_original_bucket_idx] -= (
                actual_batch_size
            )
            num_remaining_total -= actual_batch_size

            remaining_in_tier = remaining_counts_per_bucket_tier[
                chosen_original_bucket_idx
            ][chosen_tier_idx]

            # drop if not enough for batch
            if min_batch_size > 0 and 0 < remaining_in_tier < min_batch_size:
                num_remaining_total -= remaining_in_tier
                remaining_samples_per_bucket[chosen_original_bucket_idx] -= (
                    remaining_in_tier
                )

                remaining_counts_per_bucket_tier[chosen_original_bucket_idx][
                    chosen_tier_idx
                ] = 0
                remaining_indices_per_bucket_tier[chosen_original_bucket_idx][
                    chosen_tier_idx
                ].clear()

            yield batch_indices

            # Cleanup
            if remaining_samples_per_bucket[chosen_original_bucket_idx] == 0:
                current_active_buckets.pop(chosen_active_bucket_idx_in_list)

    def __len__(self) -> int:
        """
        Returns an estimated number of batches per epoch for this rank.
        """
        if self.num_samples == 0:
            return 0

        avg_bs_this_rank = self.base_batch_size
        return (self.num_samples + avg_bs_this_rank - 1) // max(1, avg_bs_this_rank)

    def set_epoch(self, epoch: int):
        """
        Updates the current epoch index to vary pseudorandom shuffling.
        """
        self.epoch = epoch
        self.generator.manual_seed(self.seed + self.rank + epoch)

class RAMTokenTierBatchSampler(BaseBatchSampler):
    """
    Map-style batch sampler mirroring StreamingTokenTierBatchSampler.
    Operates over in-RAM dataset indices, applying dynamic batch sizing
    via length penalty, strict tier alignment, and aesthetic curriculum.
    """

    AESTHETIC_TIER_MAP = AESTHETIC_TIER_MAP

    def __init__(
        self,
        dataset: "RAMCachedDataset",
        base_batch_size: int,
        tier_lengths: Optional[List[int]] = None,
        base_sequence_length: Optional[int] = None,
        length_penalty_power: float = 0.0,
        drop_last: bool = False,
        seed: Optional[int] = None,
        world_size: int = 1,
        rank: int = 0,
        aesthetic_curriculum: bool = True,
        min_batch_size: Optional[int] = None,
    ):
        super().__init__(
            dataset_len=len(dataset),
            dataset_ref=dataset,
            world_size=world_size,
            rank=rank,
            seed=seed,
            drop_last=drop_last,
        )
        self.dataset = dataset
        self.base_batch_size = base_batch_size
        self.tier_lengths = sorted(tier_lengths or TIER_LENGTHS)
        self.num_tiers = len(self.tier_lengths)
        self.tier_to_idx = {length: i for i, length in enumerate(self.tier_lengths)}
        self.base_sequence_length = base_sequence_length or self.tier_lengths[0]
        self.length_penalty_power = length_penalty_power
        self.aesthetic_curriculum = aesthetic_curriculum
        self.min_batch_size = (
            min_batch_size
            if min_batch_size is not None
            else max(1, base_batch_size // 3)
        )
        self.num_aesthetic_tiers = 5
        self.samples_yielded = 0
        self.total_samples = len(self.indices)

        # Group indices assigned to this rank into [tier_idx][aesthetic_tier_idx]
        self.bins: List[List[List[int]]] = [
            [[] for _ in range(self.num_aesthetic_tiers)]
            for _ in range(self.num_tiers)
        ]
        for global_idx in self.indices:
            seq_len = self.dataset.tiers[global_idx]
            tier_idx = self._determine_tier_idx(seq_len)
            aes_raw = self.dataset.aes_tiers[global_idx]
            aes_idx = self.AESTHETIC_TIER_MAP.get(aes_raw, 4)
            self.bins[tier_idx][aes_idx].append(global_idx)

    def _determine_tier_idx(self, seq_len: int) -> int:
        if seq_len in self.tier_to_idx:
            return self.tier_to_idx[seq_len]
        for idx, max_len in enumerate(self.tier_lengths):
            if seq_len <= max_len:
                return idx
        return self.num_tiers - 1

    def _calculate_batch_size(self, tier_idx: int) -> int:
        if self.length_penalty_power <= 0.0:
            return self.base_batch_size
        tier_length = self.tier_lengths[tier_idx]
        scale = (
            self.base_sequence_length / max(tier_length, 1)
        ) ** self.length_penalty_power
        return max(1, int(round(self.base_batch_size * min(scale, 1.0))))

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.samples_yielded = 0
        self.generator.manual_seed(self.seed + self.rank + epoch)

    def __iter__(self) -> Iterator[List[int]]:
        remaining = [
            [list(aes_list) for aes_list in tier_list]
            for tier_list in self.bins
        ]
        tier_counts = [0] * self.num_tiers
        aes_counts = [
            [0] * self.num_aesthetic_tiers for _ in range(self.num_tiers)
        ]

        # Shuffle indices within bins and initialize counters
        for t_idx in range(self.num_tiers):
            for a_idx in range(self.num_aesthetic_tiers):
                lst = remaining[t_idx][a_idx]
                if lst:
                    perm = torch.randperm(
                        len(lst), generator=self.generator
                    ).tolist()
                    remaining[t_idx][a_idx] = [lst[i] for i in perm]
                count = len(remaining[t_idx][a_idx])
                aes_counts[t_idx][a_idx] = count
                tier_counts[t_idx] += count

        while True:
            ready_tiers = [
                i
                for i, count in enumerate(tier_counts)
                if count >= self._calculate_batch_size(i)
            ]
            if not ready_tiers:
                break

            ready_weights = torch.tensor(
                [tier_counts[i] for i in ready_tiers], dtype=torch.float32
            )
            slot = torch.multinomial(
                ready_weights, 1, generator=self.generator
            ).item()
            chosen_tier = ready_tiers[slot]
            target_bs = self._calculate_batch_size(chosen_tier)

            batch_indices = []
            while len(batch_indices) < target_bs:
                avail_aes = [
                    a
                    for a in range(self.num_aesthetic_tiers)
                    if aes_counts[chosen_tier][a] > 0
                ]
                if not avail_aes:
                    break

                if self.aesthetic_curriculum:
                    prog = min(
                        1.0,
                        self.samples_yielded / max(1, self.total_samples),
                    )
                    p_simple = (1.0 - prog) * 0.4 + prog * 0.25
                    p_complex = (1.0 - prog) * 0.1 + prog * 0.25
                    weights = [
                        p_simple if a in (1, 2) else p_complex for a in avail_aes
                    ]
                else:
                    weights = [
                        float(aes_counts[chosen_tier][a]) for a in avail_aes
                    ]

                aes_tensor = torch.tensor(weights, dtype=torch.float32)
                aes_slot = torch.multinomial(
                    aes_tensor, 1, generator=self.generator
                ).item()
                chosen_aes = avail_aes[aes_slot]

                num_needed = target_bs - len(batch_indices)
                num_avail = aes_counts[chosen_tier][chosen_aes]
                take = min(num_needed, num_avail)

                src = remaining[chosen_tier][chosen_aes]
                for _ in range(take):
                    batch_indices.append(src.pop())

                aes_counts[chosen_tier][chosen_aes] -= take

            # Fill remainder if needed
            if len(batch_indices) < target_bs:
                for a in range(self.num_aesthetic_tiers):
                    src = remaining[chosen_tier][a]
                    while src and len(batch_indices) < target_bs:
                        batch_indices.append(src.pop())
                        aes_counts[chosen_tier][a] -= 1

            tier_counts[chosen_tier] -= len(batch_indices)
            self.samples_yielded += len(batch_indices)

            yield batch_indices

    def __len__(self) -> int:
        return (self.num_samples + self.base_batch_size - 1) // max(
            1, self.base_batch_size
        )
