#!/bin/bash

# Create directories if they don't exist
mkdir -p models outputs uploads

# Check if .env file exists, create from example if not
if [ ! -f .env ]; then
  echo "Creating .env file from .env.example..."
  cp .env.example .env
  echo "Please edit .env to set your Google API key if needed."
fi

# Check if models exist
if [ ! -d models/transformer ] || [ ! -f models/pytorch_lora_weights.safetensors ]; then
  echo "Model files not found. Downloading from Hugging Face..."
  python download_model.py --use_hf
fi

# Build and run Docker container
echo "Building and starting Docker container..."
docker-compose up -d

echo "API is running at http://localhost:8000"
echo "Swagger docs available at http://localhost:8000/docs"
echo ""
echo "To test the API, you can use the client script:"
echo "python client.py --image path/to/image.jpg --prompt \"your text prompt\""