#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
export WANDB_API_KEY="YOUR-API-KEY"
export HUGGING_FACE_HUB_TOKEN="YOUR-HF-TOKEN"
REPOSITORY_URL="https://codeberg.org/aipracticecafe/asuka-fm.git"
DATASET_REPO_ID="YOUR-LANTENS-REPO"

if [ -z "$WANDB_API_KEY" ]; then
  echo "Error: WANDB_API_KEY environment variable is not set."
  exit 1
fi

if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
  echo "Error: HUGGING_FACE_HUB_TOKEN environment variable is not set."
  exit 1
fi

echo "Cloning the training repository..."
git clone "${REPOSITORY_URL}"

cd asuka-fm/

echo "Installing Python packages..."
pip install -r requirements.txt
# install fa
# pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.11/flash_attn-2.8.3+cu128torch2.8-cp312-cp312-linux_x86_64.whl

echo "Logging into services..."
wandb login "$WANDB_API_KEY"
huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN"

echo "Downloading model checkpoints..."
mkdir -p models/vae/ models/unet/ models/clip/

# Download VAE weights
wget https://huggingface.co/NovelAI/nai-anime-v2/resolve/main/vae/diffusion_pytorch_model.safetensors \
  -P models/vae/

# Download UNet weights
wget https://huggingface.co/NovelAI/nai-anime-v2/resolve/main/unet/diffusion_pytorch_model.safetensors \
  -P models/unet/

# Download Text Encoder weights
wget https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/main/text_encoder/model.safetensors \
  -P models/clip/

echo "Downloading pre-computed latents dataset..."
huggingface-cli download \
  --repo-type dataset \
  "$DATASET_REPO_ID" \
  --local-dir data/

echo "Setup complete. The environment is ready for training."

