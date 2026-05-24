import sys
import os
import argparse
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trainer import Trainer


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

    # The Trainer handles DDP initialization, model loading, and the training loop
    trainer = Trainer(cfg)

    trainer.fit()


if __name__ == "__main__":
    main()
