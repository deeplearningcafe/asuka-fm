import torch
import warnings
import logging
from torch.utils.data import DataLoader
import torch.distributed as dist
from huggingface_hub.utils import _http
from src.data.dataset import (
    H5LatentDataset,
    _close_h5_handles_worker,
    WORKER_H5_HANDLES,
)
from src.data.batch_sampler import StreamingTokenTierBatchSampler, BucketBatchSampler
from src.data.streaming_dataset import StreamingImageDataset

def worker_init_fn(worker_id: int) -> None:
    """Isolates HTTP clients and resets connection pools per worker."""
    try:
        _http.reset_sessions()
    except (ImportError, AttributeError):
        pass

def create_dataloader(cfg, rank, tokenizer=None) -> DataLoader:
    """Instantiates DataLoader based on dataset_type (h5 vs streaming)."""
    dataset_type = getattr(cfg.data, "dataset_type", "h5")

    if dist.is_available() and dist.is_initialized():
        runtime_world_size = dist.get_world_size()
    else:
        runtime_world_size = 1

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
            world_size=runtime_world_size,
            low_ram=getattr(cfg.data, "low_ram", False),
        )

        tier_lengths = cfg.data.get(
            "tier_lengths", getattr(dataset, "length_tiers", [77, 152, 227])
        )
        # 10K uses a lot of ram
        default_buf = min(1024, dataset.samples_per_shard)
        buffer_size = cfg.data.get("buffer_size", default_buf)

        batched_stream = StreamingTokenTierBatchSampler(
            dataset=dataset,
            base_batch_size=cfg.train.batch_size,
            tier_lengths=tier_lengths,
            base_sequence_length=cfg.data.get("base_sequence_length", 77),
            length_penalty_power=cfg.data.get("length_penalty_power", 0.0),
            drop_last=cfg.data.get("drop_last", False),
            seed=cfg.train.seed,
            world_size=runtime_world_size,
            rank=rank,
            aesthetic_curriculum=cfg.data.get("aesthetic_curriculum", True),
            buffer_size=buffer_size,
        )

        return DataLoader(
            batched_stream,
            batch_size=None,
            num_workers=cfg.train.get("num_workers", 4),
            pin_memory=True,
            persistent_workers=True if cfg.train.get("num_workers", 4) > 0 else False,
            prefetch_factor=(
                cfg.train.get("prefetch_factor", 2)
                if cfg.train.get("num_workers", 4) > 0
                else None
            ),
            worker_init_fn=worker_init_fn,
        )

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
        world_size=runtime_world_size,
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
    logging.info(f"Initializing worker {worker_id}")
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
        logging.info("Running in main process, worker_init_fn called.")


def cleanup_h5_handles():
    """
    Iterates over and terminates all remaining H5 file handles to prevent
    unmanaged system level resource leakage.
    """
    global WORKER_H5_HANDLES
    logging.info("Attempting to clean up H5 handles...")
    worker_ids = list(WORKER_H5_HANDLES.keys())
    for worker_id in worker_ids:
        _close_h5_handles_worker(worker_id)
    logging.info(f"Cleanup finished for workers: {worker_ids}")
