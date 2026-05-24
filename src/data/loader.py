import torch
import warnings
from torch.utils.data import DataLoader
from src.data.dataset import (
    H5LatentDataset,
    _close_h5_handles_worker,
    WORKER_H5_HANDLES,
)
from src.data.batch_sampler import BucketBatchSampler


def create_dataloader(cfg, rank) -> DataLoader:
    """
    Creates and configures a PyTorch DataLoader utilizing aspect-ratio
    bucketing and custom sequence length / aesthetic scheduling.
    """
    metadata_path = f"{cfg.data.h5_path}/metadata.json"
    base_area = 64 * 96
    length_penalty = 0.1
    drop_last = False
    pin_memory = True
    persistent_workers = True
    initial_epoch_focus_low_res = 2
    low_res_focus_factor = 3.0
    low_res_area_percentile = 0.4

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
