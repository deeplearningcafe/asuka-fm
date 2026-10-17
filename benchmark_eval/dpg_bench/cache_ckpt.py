import argparse
import logging
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

DEFAULT_HF_REPO = "xingjianleng/mplug_visual-question-answering_coco_large_en"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download mPLUG VQA checkpoint from Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_HF_REPO,
        help="Hugging Face repo containing mPLUG weights.",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default="models",
        help="Target directory to save model weights (default: HF cache).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.info(f"Downloading checkpoint from Hugging Face: {args.repo_id}")

    local_model_path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
    )
    logging.info(f"Checkpoint stored locally at: {local_model_path}")

    logging.info("Validating local checkpoint loading with ModelScope...")
    pipeline(
        Tasks.visual_question_answering,
        model=local_model_path,
        device="cpu",
    )
    logging.info("mPLUG model loaded and verified successfully.")


if __name__ == "__main__":
    main()
