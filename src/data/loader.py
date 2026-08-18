import torch
import warnings
from torch.utils.data import DataLoader
from src.data.dataset import (
    H5LatentDataset,
    _close_h5_handles_worker,
    WORKER_H5_HANDLES,
)
from src.data.batch_sampler import StreamingTokenTierBatchSampler, BucketBatchSampler
from src.data.streaming_dataset import StreamingImageDataset


def create_dataloader(cfg, rank, tokenizer=None) -> DataLoader:
    """Instantiates DataLoader based on dataset_type (h5 vs streaming)."""
    dataset_type = getattr(cfg.data, "dataset_type", "h5")

    if dataset_type == "streaming":
        dataset = StreamingImageDataset(
            dataset_name=cfg.data.streaming_dataset_name,
            dataset_path=cfg.data.get("dataset_path", None),
            resolution=cfg.data.get("resolution", 512),
            patch_size=cfg.data.get("patch_size", 2),
            vae_downsample_factor=cfg.data.get("vae_downsample_factor", 8),
            max_seq_len=cfg.data.get("max_seq_len", 256),
            tokenizer=tokenizer,
            cfg_dropout_prob=getattr(cfg.train, "cfg_dropout_prob", 0.0),
            tag_dropout_prob=getattr(cfg.data, "tag_dropout", 0.0),
            shuffle_tags=getattr(cfg.data, "shuffle_tags", True),
            rank=rank,
            world_size=getattr(cfg.train, "world_size", 1),
        )

        tier_lengths = cfg.data.get(
            "tier_lengths", getattr(dataset, "length_tiers", [77, 152, 227])
        )
        batched_stream = StreamingTokenTierBatchSampler(
            dataset=dataset,
            base_batch_size=cfg.train.batch_size,
            tier_lengths=tier_lengths,
            base_sequence_length=cfg.data.get("base_sequence_length", 77),
            length_penalty_power=cfg.data.get("length_penalty_power", 0.0),
            drop_last=cfg.data.get("drop_last", False),
            seed=cfg.train.seed,
            world_size=getattr(cfg.train, "world_size", 1),
            rank=rank,
            aesthetic_curriculum=cfg.data.get("aesthetic_curriculum", True),
            buffer_size=cfg.data.get("buffer_size", dataset.samples_per_shard),
        )

        dataloader = DataLoader(
            batched_stream,
            batch_size=None,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.get("pin_memory", True),
            prefetch_factor=(
                cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None
            ),
            persistent_workers=(
                cfg.data.get("persistent_workers", True)
                if cfg.data.num_workers > 0
                else False
            ),
        )
        return dataloader

    metadata_path = f"{cfg.data.h5_path}/metadata.json"
    base_area = cfg.data.get("base_resolution_area", 64 * 64)
    length_penalty = cfg.data.get("length_penalty_power", 0.1)
    drop_last = cfg.data.get("drop_last", False)
    pin_memory = cfg.data.get("pin_memory", True)
    persistent_workers = cfg.data.get("persistent_workers", True)
    initial_epoch_focus_low_res = cfg.data.get("initial_epoch_focus_low_res", 0)
    low_res_focus_factor = cfg.data.get("low_res_focus_factor", 1.0)
    low_res_area_percentile = cfg.data.get("low_res_area_percentile", 0.33)

    dataset = H5LatentDataset(
        metadata_path=metadata_path,
        h5_root_dir=cfg.data.h5_path,
        load_into_ram=getattr(cfg.data, "load_into_ram", False),
        tag_dropout=getattr(cfg.data, "tag_dropout", 0.0),
    )

    batch_sampler = BucketBatchSampler(
        dataset=dataset,
        base_batch_size=cfg.train.batch_size,
        length_penalty_power=length_penalty,
        base_resolution_area=base_area,
        drop_last=drop_last,
        seed=cfg.train.seed,
        world_size=cfg.train.world_size,
        rank=rank,
        initial_epoch_focus_low_res=initial_epoch_focus_low_res,
        low_res_focus_factor=low_res_focus_factor,
        low_res_area_percentile=low_res_area_percentile,
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=(
            cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None
        ),
        persistent_workers=(persistent_workers if cfg.data.num_workers > 0 else False),
        worker_init_fn=worker_init_fn,
    )

    return dataloader


def worker_init_fn(worker_id: int):
    """
    Initializes individual worker processes, invoking internal state setup
    for worker-exclusive H5 handle generation.
    """
    print(f"Initializing worker {worker_id}")
    worker_info = torch.utils.data.get_worker_info()
    if worker_info:
        dataset = worker_info.dataset
        if isinstance(dataset, H5LatentDataset):
            dataset._initialize_worker()
        else:
            warnings.warn(
                f"Worker {worker_id} received unexpected dataset type: {type(dataset)}"
            )
    else:
        print("Running in main process, worker_init_fn called.")


def cleanup_h5_handles():
    """
    Iterates over and terminates all remaining H5 file handles to prevent
    unmanaged system level resource leakage.
    """
    global WORKER_H5_HANDLES
    print("Attempting to clean up H5 handles...")
    worker_ids = list(WORKER_H5_HANDLES.keys())
    for worker_id in worker_ids:
        _close_h5_handles_worker(worker_id)
    print(f"Cleanup finished for workers: {worker_ids}")
