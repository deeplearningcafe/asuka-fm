import torch
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from typing import List, Dict, Iterator, Optional
import warnings

from src.data.dataset import (
    H5LatentDataset,
    TIER_LENGTHS,
    LENGTH_TO_TIER_IDX,
    NUM_TIERS,
)


class BucketBatchSampler(Sampler[List[int]]):
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
        super().__init__(dataset)
        if world_size > 1:
            # DistributedSampler logic for subsetting indices per rank
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=seed
            )
            self.num_samples_total = len(dataset)
            self.indices = list(sampler)
            self.num_samples = len(self.indices)
            print(
                f"[Rank {rank}] DDP Sampler: Using {self.num_samples} / "
                f"{self.num_samples_total} samples."
            )
        else:
            self.indices = list(range(len(dataset)))
            self.num_samples = len(dataset)
            self.num_samples_total = self.num_samples

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
        self.drop_last = drop_last
        self.generator = torch.Generator()
        self.seed = seed if seed is not None else torch.seed()
        self.generator.manual_seed(self.seed + rank)
        self.max_batch_size_ratio = max_batch_size_ratio
        self.world_size = world_size
        self.rank = rank
        self.epoch = 0

        if self.num_samples == 0:
            raise ValueError("Dataset is empty.")

        self.buckets_idx_list = [bucket["bucket_idx"] for bucket in dataset.bucket_info]
        self.num_buckets = len(dataset.bucket_info)
        if self.num_buckets == 0:
            raise ValueError("Dataset metadata lacks bucket information.")

        self.num_tiers = NUM_TIERS
        self.tier_lengths = TIER_LENGTHS
        self.num_aesthetic_tiers = 5
        AESTHETIC_TIER_MAP = {-1: 4, 0: 0, 1: 1, 2: 2, 3: 3}

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
