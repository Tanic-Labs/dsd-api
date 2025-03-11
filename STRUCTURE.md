# Project Structure

This file provides an overview of the directory structure and files in the Diffusion Self-Distillation API project.

```
dsd-api/
│
├── main.py                 # Main FastAPI application
├── client.py               # Client script for testing the API
├── download_model.py       # Script to download model files
├── run_docker.sh           # Script to run the Docker container
│
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment variables
├── Dockerfile              # Docker build instructions
├── docker-compose.yml      # Docker Compose configuration
├── .dockerignore           # Files to ignore in Docker context
│
├── README.md               # Project documentation
├── STRUCTURE.md            # This file
│
├── models/                 # [Directory] Stores model files
│   ├── transformer/        # Transformer model directory
│   │   ├── config.json     # Model configuration
│   │   └── diffusion_pytorch_model.safetensors  # Model weights
│   └── pytorch_lora_weights.safetensors  # LoRA weights
│
├── outputs/                # [Directory] Stores generated images
└── uploads/                # [Directory] Stores uploaded images
```

## Key Files

- **main.py**: The FastAPI application that provides the REST API endpoints
- **client.py**: A Python script to test the API functionality
- **download_model.py**: Utility script to download the model files from Hugging Face or Google Drive
- **run_docker.sh**: Helper script to set up and run the Docker container
- **Dockerfile**: Instructions for building the Docker image
- **docker-compose.yml**: Configuration for Docker Compose deployment

## Directories

- **models/**: Contains the DSD model files
- **outputs/**: Stores generated images from the API
- **uploads/**: Temporarily stores uploaded images for processing

## Environment Variables

Environment variables can be configured in the `.env` file:

- `MODEL_PATH`: Path to the transformer model directory
- `LORA_PATH`: Path to the LoRA weights file
- `OUTPUT_DIR`: Directory for storing generated images
- `UPLOAD_DIR`: Directory for storing uploaded images
- `GOOGLE_API_KEY`: Google Gemini API key for prompt enhancement