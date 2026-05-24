import json
import h5py
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings
import gc
from src.data.utils import shuffle_prompt_token_ids

# Worker-specific HDF5 file handles to avoid resource contention
WORKER_H5_HANDLES = {}

TIER_LENGTHS = [77, 152, 227]
LENGTH_TO_TIER_IDX = {length: i for i, length in enumerate(TIER_LENGTHS)}
NUM_TIERS = len(TIER_LENGTHS)
END_OF_TEXT_ID = 49407


def _init_h5_handles_worker(
    worker_id: int, h5_root_dir: Path, shard_filenames: List[str]
):
    """
    Initializes and caches H5 file handles for a given worker ID.
    """
    global WORKER_H5_HANDLES
    if worker_id not in WORKER_H5_HANDLES:
        WORKER_H5_HANDLES[worker_id] = {}
        for filename in shard_filenames:
            shard_path = h5_root_dir / filename
            if shard_path.exists():
                try:
                    handle = h5py.File(shard_path, "r", libver="latest", swmr=False)
                    WORKER_H5_HANDLES[worker_id][filename] = handle
                except Exception as e:
                    warnings.warn(
                        f"[Worker {worker_id}] Error opening H5 file "
                        f"{shard_path}: {e}. It will be skipped."
                    )
                    WORKER_H5_HANDLES[worker_id][filename] = None
            else:
                warnings.warn(
                    f"[Worker {worker_id}] Shard file not found: "
                    f"{shard_path}. It will be skipped."
                )
                WORKER_H5_HANDLES[worker_id][filename] = None


def _close_h5_handles_worker(worker_id: int):
    """
    Closes and removes cached H5 file handles for a given worker ID.
    """
    global WORKER_H5_HANDLES
    if worker_id in WORKER_H5_HANDLES:
        for filename, handle in WORKER_H5_HANDLES[worker_id].items():
            if handle is not None:
                try:
                    handle.close()
                except Exception as e:
                    warnings.warn(
                        f"[Worker {worker_id}] Error closing handle for {filename}: {e}"
                    )
        del WORKER_H5_HANDLES[worker_id]
        gc.collect()


def _get_h5_handle_worker(worker_id: int, shard_filename: str) -> Optional[h5py.File]:
    """
    Retrieves the cached H5 handle for a specific worker and shard filename.
    """
    global WORKER_H5_HANDLES
    if worker_id not in WORKER_H5_HANDLES:
        warnings.warn(
            f"H5 handles not initialized for worker {worker_id}. "
            f"Attempting to access {shard_filename} will likely fail."
        )
        return None
    return WORKER_H5_HANDLES.get(worker_id, {}).get(shard_filename, None)


class H5LatentDataset(Dataset):
    """
    Dataset for sharded H5 files containing precomputed text embeddings,
    tokenized captions, and VAE latents.
    """

    def __init__(
        self,
        metadata_path: str | Path,
        h5_root_dir: str | Path,
        load_into_ram: bool = False,
        tag_dropout: float = 0.0,
    ):
        """
        Args:
            metadata_path: Path to the metadata.json file.
            h5_root_dir: Root directory containing the H5 shard files.
            load_into_ram: Whether to load the entire dataset into RAM.
            tag_dropout: Probability of dropping a general tag.
        """
        self.metadata_path = Path(metadata_path)
        self.h5_root_dir = Path(h5_root_dir)
        self._h5_handles: Dict[str, Optional[h5py.File]] = {}
        self.load_into_ram = load_into_ram
        self.tag_dropout = tag_dropout
        self.ram_cache = {}

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        if not self.h5_root_dir.is_dir():
            raise NotADirectoryError(f"H5 root directory not found: {self.h5_root_dir}")

        print(f"Loading metadata from {self.metadata_path}...")
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.sample_mapping = self.metadata.get("sample_mapping", [])
        if not self.sample_mapping:
            raise ValueError("Metadata file does not contain 'sample_mapping'.")

        self.bucket_info = self.metadata.get("bucket_info", [])
        if not self.bucket_info:
            warnings.warn(
                "Metadata file does not contain 'bucket_info'. "
                "Bucket-specific info might be unavailable."
            )

        self.all_shard_filenames = sorted(
            list(set(s["shard_file"] for s in self.sample_mapping if "shard_file" in s))
        )

        self.is_cache_text_embeds = self.metadata["dataset_info"].get(
            "cache_text_embeds", False
        )
        self.is_store_tokenized_captions = self.metadata["dataset_info"].get(
            "store_tokenized_captions", False
        )

        if not self.is_cache_text_embeds and not self.is_store_tokenized_captions:
            raise ValueError("Not cache embeds nor input_ids")

        print(
            f"Found metadata for {len(self.sample_mapping)} samples across "
            f"{len(self.bucket_info)} buckets, referencing "
            f"{len(self.all_shard_filenames)} shard files."
            f"Using Cache text embeddings: {self.is_cache_text_embeds}"
            f"Using Store tokenized captions: {self.is_store_tokenized_captions}"
        )

        if self.load_into_ram:
            self._load_data_into_ram()

        self.worker_id = 0

    def _load_data_into_ram(self):
        """
        Loads and caches all H5 shards directly into memory.
        """
        print("Loading H5 dataset into RAM. This may take a while...")
        for shard_filename in self.all_shard_filenames:
            shard_path = self.h5_root_dir / shard_filename
            if not shard_path.exists():
                warnings.warn(f"Shard not found: {shard_path}")
                continue

            try:
                with h5py.File(shard_path, "r", libver="latest", swmr=False) as f:
                    self.ram_cache[shard_filename] = self._recursively_load_group(f)
            except Exception as e:
                warnings.warn(f"Error loading {shard_path} into RAM: {e}")

    def _recursively_load_group(self, group: h5py.Group) -> dict:
        """
        Recursively transforms an HDF5 group into a nested dictionary of arrays.
        """
        data = {}
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                data[key] = item[:]
            elif isinstance(item, h5py.Group):
                data[key] = self._recursively_load_group(item)
        return data

    def _initialize_worker(self):
        """
        Initializes worker identity and registers H5 file handles.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.worker_id = worker_info.id
        else:
            self.worker_id = 0

        # H5 handles if not loading into RAM
        if not self.load_into_ram:
            _init_h5_handles_worker(
                self.worker_id, self.h5_root_dir, self.all_shard_filenames
            )

    def __len__(self) -> int:
        """
        Returns the total number of samples.
        """
        return len(self.sample_mapping)

    def _process_data(
        self,
        latent_np,
        tag_weight_np,
        te_np=None,
        prefix_np=None,
        general_np=None,
        suffix_np=None,
        idx=None,
        bucket_idx=None,
        tier=None,
        booru_id=None,
        shard_file=None,
    ):
        """
        Formats numpy data arrays into tensors and applies augmentations.
        """

        latent_tensor = torch.from_numpy(latent_np)

        tag_weight_tensor = torch.from_numpy(tag_weight_np)

        if self.is_cache_text_embeds:
            te_tensor = torch.from_numpy(te_np)

            return latent_tensor, te_tensor, tag_weight_tensor, None

        elif self.is_store_tokenized_captions:
            general_tokens_list = general_np.tolist()

            max_prompt_length = TIER_LENGTHS[-1]
            len_no_general = len(prefix_np) + len(suffix_np)
            current_len = len_no_general + len(general_np)

            shuffled_general_tokens = shuffle_prompt_token_ids(
                general_tokens_list,
                drop_prob=self.tag_dropout,
                prompt_len=current_len,
                include_special_tokens=False,
            )

            prefix_tensor = torch.from_numpy(prefix_np).to(torch.long)
            shuffled_general_tensor = torch.tensor(
                shuffled_general_tokens, dtype=torch.long
            )
            suffix_tensor = torch.from_numpy(suffix_np).to(torch.long)

            final_len = (
                len(prefix_tensor) + len(shuffled_general_tensor) + len(suffix_tensor)
            )
            padding = torch.empty(0, dtype=torch.long)
            if final_len > max_prompt_length:
                general_tokens_len = max_prompt_length - len_no_general
                shuffled_general_tensor = shuffled_general_tensor[:general_tokens_len]
            elif final_len < max_prompt_length:
                padding_needed = max_prompt_length - final_len
                padding = torch.full(
                    (padding_needed,), END_OF_TEXT_ID, dtype=torch.long
                )
            # concat with empty tensor ignores it
            reconstructed_tokens = torch.cat(
                (prefix_tensor, shuffled_general_tensor, suffix_tensor, padding),
                dim=0,
            )
            if reconstructed_tokens.shape[-1] > 227:
                warnings.warn(
                    f"[Worker {self.worker_id}] prompt length > 227 sample {idx} "
                    f"(bkt {bucket_idx}, tier {tier}, "
                    f"booru id {booru_id}) in {shard_file}."
                )
                return None

            not_pad_mask = reconstructed_tokens != END_OF_TEXT_ID

            # moves mask from the last valid token onto the first padding token
            shifted_mask = torch.roll(not_pad_mask, shifts=1, dims=0)

            # first token (BOS) is always valid
            shifted_mask[0] = True

            attention_mask = not_pad_mask | shifted_mask

            return (
                latent_tensor,
                reconstructed_tokens,
                tag_weight_tensor,
                attention_mask,
            )

    def _read_from_disk(self, sample_meta: dict, idx: int) -> Optional[Tuple]:
        """
        Fetches and processes a single data record from disk H5 files.
        """
        shard_file = sample_meta.get("shard_file")
        bucket_idx = sample_meta.get("bucket_idx")
        tier = sample_meta.get("tier")
        idx_in_tier = sample_meta.get("idx_in_tier")
        booru_id = sample_meta.get("booru_id")

        h5_file = _get_h5_handle_worker(self.worker_id, shard_file)
        if h5_file is None:
            return None

        try:
            b_path = f"bucket_{bucket_idx}/tier_{tier}"
            latent_np = h5_file[f"{b_path}/latents"][idx_in_tier]
            tag_weight_np = h5_file[f"{b_path}/tag_weight"][idx_in_tier]

            if self.is_cache_text_embeds:
                te_np = h5_file[f"{b_path}/text_embeddings"][idx_in_tier]
                return self._process_data(latent_np, tag_weight_np, te_np=te_np)
            elif self.is_store_tokenized_captions:
                prefix_np = h5_file[f"{b_path}/prefix_tokens"][idx_in_tier]
                general_np = h5_file[f"{b_path}/general_tokens"][idx_in_tier]
                suffix_np = h5_file[f"{b_path}/suffix_tokens"][idx_in_tier]
                return self._process_data(
                    latent_np,
                    tag_weight_np,
                    prefix_np=prefix_np,
                    general_np=general_np,
                    suffix_np=suffix_np,
                    idx=idx,
                    bucket_idx=bucket_idx,
                    tier=tier,
                    booru_id=booru_id,
                    shard_file=shard_file,
                )
        except Exception as e:
            warnings.warn(
                f"[Worker {self.worker_id}] Error loading sample {idx} "
                f"from {shard_file}: {e}"
            )
            return None

    def _read_from_ram(self, sample_meta: dict, idx: int) -> Optional[Tuple]:
        """
        Fetches and processes a single data record from RAM-cached datasets.
        """
        shard_file = sample_meta.get("shard_file")
        bucket_idx = sample_meta.get("bucket_idx")
        tier = sample_meta.get("tier")
        idx_in_tier = sample_meta.get("idx_in_tier")
        booru_id = sample_meta.get("booru_id")

        shard_data = self.ram_cache.get(shard_file)
        if shard_data is None:
            return None

        try:
            tier_data = shard_data[f"bucket_{bucket_idx}"][f"tier_{tier}"]
            latent_np = tier_data["latents"][idx_in_tier]
            tag_weight_np = tier_data["tag_weight"][idx_in_tier]

            if self.is_cache_text_embeds:
                te_np = tier_data["text_embeddings"][idx_in_tier]
                return self._process_data(latent_np, tag_weight_np, te_np=te_np)
            elif self.is_store_tokenized_captions:
                prefix_np = tier_data["prefix_tokens"][idx_in_tier]
                general_np = tier_data["general_tokens"][idx_in_tier]
                suffix_np = tier_data["suffix_tokens"][idx_in_tier]
                return self._process_data(
                    latent_np,
                    tag_weight_np,
                    prefix_np=prefix_np,
                    general_np=general_np,
                    suffix_np=suffix_np,
                    idx=idx,
                    bucket_idx=bucket_idx,
                    tier=tier,
                    booru_id=booru_id,
                    shard_file=shard_file,
                )
        except Exception as e:
            warnings.warn(
                f"[Worker {self.worker_id}] Error loading sample {idx} "
                f"from RAM cache ({shard_file}): {e}"
            )
            return None

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Fetches a dataset element by list index.
        """
        if not 0 <= idx < len(self.sample_mapping):
            warnings.warn(f"Index {idx} out of bounds.")
            return None

        sample_meta = self.sample_mapping[idx]
        if None in [
            sample_meta.get("shard_file"),
            sample_meta.get("bucket_idx"),
            sample_meta.get("tier"),
            sample_meta.get("idx_in_tier"),
        ]:
            warnings.warn(f"Sample {idx} has incomplete metadata.")
            return None

        if self.load_into_ram:
            return self._read_from_ram(sample_meta, idx)
        else:
            return self._read_from_disk(sample_meta, idx)

    def close_handles(self):
        """
        Closes open HDF5 file handles associated with the current worker.
        """
        _close_h5_handles_worker(self.worker_id)
