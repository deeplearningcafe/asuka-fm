import sys
import os
import argparse
from omegaconf import OmegaConf
from datetime import datetime
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trainer import Trainer
from src.utils.logging_utils import Logger


def main():
    parser = argparse.ArgumentParser(description="Train Flow Matching/DDPM Model")
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
    save_dir = os.path.join("results", "train", timestamp)
    Logger.setup_logging(
        save_dir=save_dir,
        logging_name=f"{model_type}_loss_{cfg.train['objective']}",
    )
    logging.info(cfg)
    # The Trainer handles DDP initialization, model loading, and the training loop
    trainer = Trainer(cfg)

    trainer.fit(timestamp=timestamp)


if __name__ == "__main__":
    main()
