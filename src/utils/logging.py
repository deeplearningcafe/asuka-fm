import wandb
from huggingface_hub import HfApi
from PIL import Image

_wandb_run = None
_hf_api = None


def init_wandb(
    project_name: str, run_name: str = None, config: dict = None, entity: str = None
):
    """
    Initializes a new W&B run.

    Args:
        project_name (str): Name of the W&B project.
        run_name (str, optional): Name of the run. Defaults to None (W&B auto-generates).
        config (dict, optional): Hyperparameters and other run configurations.
        entity (str, optional): W&B entity (username or team name).
    """
    global _wandb_run
    try:
        if wandb.run is not None:
            print("W&B run already initialized. Finishing the current one.")
            wandb.finish()

        _wandb_run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            entity=entity,
            reinit=True,
        )
        print(f"W&B run initialized: {_wandb_run.url}")
    except Exception as e:
        print(f"Error initializing W&B: {e}. W&B logging will be disabled.")
        _wandb_run = None


def log_metrics(metrics_dict: dict, step: int = None, commit: bool = False):
    """
    Logs a dictionary of metrics to W&B with improved error handling.

    Args:
        metrics_dict (dict): Dictionary of metric_name: value.
        step (int, optional): The current step (e.g., batch or epoch).
                             If None, W&B uses its internal step.
        commit (bool): Whether to commit the log. Set to False if logging
                       multiple metrics for the same step incrementally.
    """
    if _wandb_run:
        try:
            if step is not None:
                _wandb_run.log(metrics_dict, step=step, commit=commit)
            else:
                _wandb_run.log(metrics_dict, commit=commit)
        except Exception as e:
            print(f"Error logging metrics to W&B: {e}")
            print(f"Metrics: {metrics_dict}, Step: {step}")


def log_image(
    imgs: list[Image.Image],
    prompts: list[str],
    epoch_num: int,
    step: int,
    commit: bool = False,
):
    """
    Logs an image to W&B with explicit step and commit control.

    Args:
        image_key (str): Key for the image in W&B dashboard.
        image_data: The image data.
        caption (str, optional): Caption for the image.
        step (int, optional): The current step.
        commit (bool): Whether to commit the log immediately.
    """
    if _wandb_run:
        try:
            log_payload = {}

            log_payload["train/image_log_step"] = step

            for i, (img, prompt) in enumerate(zip(imgs, prompts)):
                image_key = f"epoch_samples/sample_{i}"

                w_img = wandb.Image(
                    img,
                    caption=(f"Epoch: {epoch_num} | Step: {step}\nPrompt: {prompt}"),
                )
                log_payload[image_key] = w_img

            wandb.log(log_payload, commit=commit)

            print(f"Background save of images for epoch {epoch_num} complete.")
        except Exception as e:
            print(f"Error logging image to W&B: {e}")
            print(f"Image key: {image_key}, Step: {step}")


def finish_wandb():
    """Finishes the current W&B run."""
    global _wandb_run
    if _wandb_run:
        try:
            wandb.finish()
            print("W&B run finished.")
        except Exception as e:
            print(f"Error finishing W&B run: {e}")
        finally:
            _wandb_run = None


def is_wandb_initialized():
    """Checks if W&B has been successfully initialized."""
    return _wandb_run is not None


def init_hfapi(token: str = None):
    """
    Initializes a HfApi.

    Args:
        token (str): If none is passed uses the system token after logging
    """
    global _hf_api
    try:
        _hf_api = HfApi(
            token=token,
        )
        print(f"HfApi run initialized")
    except Exception as e:
        print(f"Error initializing HfApi: {e}. HfApi logging will be disabled.")
        _hf_api = None


def log_folder(
    folder_path: str,
    repo_id: str = "",
):
    """
    Logs an image to W&B with explicit step and commit control.

    Args:
        folder_path (str): Key for the image in W&B dashboard.
        repo_id (str, optional): Caption for the image.
    """
    if _hf_api:
        try:
            if len(repo_id) < 1:
                repo_id = "edm-flow-matching"
            _hf_api.create_repo(repo_id, private=True, repo_type="model", exist_ok=True)
            _hf_api.upload_large_folder(
                repo_id=repo_id, repo_type="model", folder_path=folder_path
            )
        except Exception as e:
            print(f"Error logging folder to HfApi: {e}")


def is_hfapi_initialized():
    """Checks if HfApi has been successfully initialized."""
    return _hf_api is not None
