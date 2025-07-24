#!/bin/bash
# Download Wan2.1 I2V 14B 480P model files directly

MODEL_DIR="Wan2.1-I2V-14B-480P"
BASE_URL="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P/resolve/main"

echo "Creating model directory: $MODEL_DIR"
mkdir -p $MODEL_DIR
mkdir -p $MODEL_DIR/google/umt5-xxl
mkdir -p $MODEL_DIR/xlm-roberta-large

echo "Downloading I2V 14B 480P model files..."
echo "Total size: ~82GB"
echo ""

# Download safetensors model parts
echo "Downloading model parts (7 files, ~65GB total)..."
for i in {1..7}; do
    echo "Downloading part $i of 7..."
    wget -c -P $MODEL_DIR/ "$BASE_URL/diffusion_pytorch_model-0000$i-of-00007.safetensors"
done

# Download config files
echo "Downloading configuration files..."
wget -c -P $MODEL_DIR/ "$BASE_URL/config.json"
wget -c -P $MODEL_DIR/ "$BASE_URL/diffusion_pytorch_model.safetensors.index.json"

# Download VAE
echo "Downloading VAE model (508MB)..."
wget -c -P $MODEL_DIR/ "$BASE_URL/Wan2.1_VAE.pth"

# Download CLIP model
echo "Downloading CLIP model (4.77GB)..."
wget -c -P $MODEL_DIR/ "$BASE_URL/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"

# Download T5 encoder
echo "Downloading T5 encoder (11.4GB)..."
wget -c -P $MODEL_DIR/ "$BASE_URL/models_t5_umt5-xxl-enc-bf16.pth"

# Download tokenizer files
echo "Downloading tokenizer files..."
# Google T5 tokenizer
wget -c -P $MODEL_DIR/google/umt5-xxl/ "$BASE_URL/google/umt5-xxl/special_tokens_map.json"
wget -c -P $MODEL_DIR/google/umt5-xxl/ "$BASE_URL/google/umt5-xxl/spiece.model"
wget -c -P $MODEL_DIR/google/umt5-xxl/ "$BASE_URL/google/umt5-xxl/tokenizer.json"
wget -c -P $MODEL_DIR/google/umt5-xxl/ "$BASE_URL/google/umt5-xxl/tokenizer_config.json"

# XLM-Roberta tokenizer
wget -c -P $MODEL_DIR/xlm-roberta-large/ "$BASE_URL/xlm-roberta-large/sentencepiece.bpe.model"
wget -c -P $MODEL_DIR/xlm-roberta-large/ "$BASE_URL/xlm-roberta-large/tokenizer.json"
wget -c -P $MODEL_DIR/xlm-roberta-large/ "$BASE_URL/xlm-roberta-large/tokenizer_config.json"

echo ""
echo "Download complete! Model saved to: $MODEL_DIR/"
echo "You can now use this model with: --ckpt_dir ./$MODEL_DIR"