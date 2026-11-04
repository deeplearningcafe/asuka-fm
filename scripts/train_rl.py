import sys
import os
import argparse
from omegaconf import OmegaConf
from datetime import datetime
import logging
import torch.multiprocessing as mp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl.trainer import RLTrainer
from src.utils.logging_utils import Logger


def _clean_excepthook(exc_type, exc_value, exc_traceback):
    """Ensures exceptions are cleanly flushed to stderr and loggers."""
    import traceback

    sys.stderr.write("\n=== UNCAUGHT EXCEPTION ENCOUNTERED ===\n")
    traceback.print_exception(
        exc_type, exc_value, exc_traceback, file=sys.stderr
    )
    sys.stderr.flush()
    logging.error(
        "Uncaught exception:",
        exc_info=(exc_type, exc_value, exc_traceback),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Flow Matching DiT using Flow-GRPO"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the training configuration YAML file",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at: {args.config}")

    cfg = OmegaConf.load(args.config)
    model_type = getattr(cfg.models, "model_type", "unet")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("results", "rl", timestamp)

    Logger.setup_logging(
        save_dir=save_dir,
        logging_name=f"{model_type}_rl_flow_grpo",
    )
    logging.info(cfg)

    # Initialize RLTrainer for rollout sampling and policy updates
    trainer = RLTrainer(cfg)
    trainer.fit(timestamp=timestamp)


if __name__ == "__main__":
    sys.excepthook = _clean_excepthook

    # Raise POSIX file descriptor limit to prevent resource exhaustion
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(65536, hard if hard > 0 else 65536)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception as err:
        logging.warning(f"Could not adjust RLIMIT_NOFILE: {err}")

    # Use 'file_descriptor' strategy to prevent memory leaks in /dev/shm
    try:
        mp.set_sharing_strategy("file_descriptor")
    except RuntimeError:
        pass

    main()