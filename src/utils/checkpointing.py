import torch
import os
from safetensors.torch import save_file
import src.utils.logging as logging_utils


def save_checkpoint(
    epoch: int,
    global_step: int,
    unet: torch.nn.Module,
    text_encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_te: bool,
    hf_repo: str = None,
    base_dir: str = ".",
    train_only_output: bool = False,
    ema=None,
):
    """
    Saves a complete training checkpoint.

    This function saves the model weights, optimizer and scheduler states,
    and the current training progress (epoch and step) to a dedicated
    folder. It then uploads this folder to a specified Hugging Face repo.

    Args:
        epoch: The current epoch number.
        global_step: The total number of optimization steps completed.
        unet: The UNet model.
        text_encoder: The text encoder model.
        optimizer: The optimizer instance.
        scheduler: The learning rate scheduler instance.
        train_te: A boolean indicating if the text encoder was trained.
        hf_repo: The Hugging Face repository ID to upload to.
        base_dir: The root directory to save the checkpoint folder in.
    """
    save_dir = os.path.join(base_dir, f"epoch_{epoch}_step_{global_step}")
    checkpoint_dir = os.path.join(save_dir, f"epoch_{epoch}_step_{global_step}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Saving checkpoint to {checkpoint_dir}...")

    unet_state_dict = unet.state_dict()
    if train_only_output:
        print("Filtering state dict: Saving only output head parameters.")
        # Filter keys for 'conv_out' and 'conv_norm_out'
        keys_to_save = [
            k for k in unet_state_dict.keys() if "conv_out" in k or "conv_norm_out" in k
        ]
        unet_state_dict = {k: unet_state_dict[k] for k in keys_to_save}

    save_file(unet_state_dict, os.path.join(checkpoint_dir, "unet.safetensors"))

    if ema is not None and ema.use_ema:
        print("Saving EMA weights...")
        ema_state_dict = ema.ema_model.state_dict()
        if train_only_output:
            keys_to_save = [
                k
                for k in ema_state_dict.keys()
                if "conv_out" in k or "conv_norm_out" in k
            ]
            ema_state_dict = {k: ema_state_dict[k] for k in keys_to_save}
        save_file(ema_state_dict, os.path.join(checkpoint_dir, "unet_ema.safetensors"))

    if train_te:
        save_file(
            text_encoder.state_dict(),
            os.path.join(checkpoint_dir, "text_encoder.safetensors"),
        )

    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
    if scheduler:
        torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))

    training_state = {
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))

    print("Checkpoint saved successfully.")

    if logging_utils.is_hfapi_initialized() and hf_repo:
        print(f"Uploading checkpoint to Hugging Face repo: {hf_repo}")
        logging_utils.log_folder(save_dir, hf_repo)
        print("Upload complete.")
