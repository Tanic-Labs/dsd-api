# Diffusion Self-Distillation API

This repository contains a FastAPI-based REST API for the [Diffusion Self-Distillation](https://github.com/primecai/diffusion-self-distillation) model, which enables zero-shot customized image generation.

## Features

- REST API for generating images using the Diffusion Self-Distillation model
- Accepts images as input and generates new images based on text prompts
- Optional prompt enhancement using Google Gemini
- Asynchronous processing for lower memory devices
- Docker support for easy deployment

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU with >24GB VRAM (can run with CPU offload on lower VRAM GPUs)
- Docker and Docker Compose (optional, for containerized deployment)

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/yourusername/dsd-api.git
cd dsd-api
```

### 2. Download the model

You can download the model files from Hugging Face or Google Drive:

```bash
# Download from Hugging Face
python download_model.py --use_hf

# OR download from Google Drive
python download_model.py
```

### 3. Set up environment variables

Copy the example environment file and edit it with your settings:

```bash
cp .env.example .env
```

Edit the `.env` file to set your Google Gemini API key if you want to use the prompt enhancement feature.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000. Swagger documentation is available at http://localhost:8000/docs.

## Docker Deployment

### 1. Build and run with Docker Compose

```bash
docker-compose up -d
```

This will build the Docker image and start the API container. The API will be available at http://localhost:8000.

### 2. Or build and run manually

```bash
# Build the Docker image
docker build -t dsd-api .

# Run the container
docker run -p 8000:8000 --gpus all -v $(pwd)/models:/app/models -v $(pwd)/outputs:/app/outputs -v $(pwd)/uploads:/app/uploads dsd-api
```

## API Endpoints

### `POST /generate/`

Generate an image using the Diffusion Self-Distillation model.

**Parameters:**
- `image`: The conditioning image file (multipart/form-data)
- `text`: The text prompt
- `use_gemini`: Whether to enhance the prompt using Gemini (default: true)
- `guidance`: The guidance scale (default: 3.5)
- `i_guidance`: Image guidance scale (default: 1.0)
- `t_guidance`: Text guidance scale (default: 1.0)

**Response:**
- If the GPU has enough memory, it will respond with the generated image
- If using a lower-memory GPU or CPU, it will respond with a request ID that can be used to check the status of the generation

### `GET /result/{request_id}`

Get the result of a previously submitted generation request.

**Parameters:**
- `request_id`: The ID returned from the /generate endpoint

**Response:**
- If the generation is complete, it will respond with the generated image
- If the generation is still in progress, it will respond with a status message

### `GET /health`

Check if the API is healthy and the model is loaded.

## Example Usage

Using curl:

```bash
curl -X POST http://localhost:8000/generate/ \
  -F "image=@input.jpg" \
  -F "text=a man in a suit standing on the moon" \
  -F "use_gemini=true" \
  -F "guidance=3.5" \
  -F "i_guidance=1.0" \
  -F "t_guidance=1.0" \
  -o output.png
```

Using Python:

```python
import requests

url = "http://localhost:8000/generate/"
files = {"image": open("input.jpg", "rb")}
data = {
    "text": "a man in a suit standing on the moon",
    "use_gemini": "true",
    "guidance": "3.5",
    "i_guidance": "1.0",
    "t_guidance": "1.0"
}

response = requests.post(url, files=files, data=data)

# If the response is an image, save it
if response.headers.get("content-type") == "image/png":
    with open("output.png", "wb") as f:
        f.write(response.content)
# If the response is JSON (async processing), get the request ID
else:
    result = response.json()
    request_id = result["request_id"]
    print(f"Processing started with request ID: {request_id}")
    
    # Poll the result endpoint until the image is ready
    while True:
        result_response = requests.get(f"http://localhost:8000/result/{request_id}")
        if result_response.headers.get("content-type") == "image/png":
            with open("output.png", "wb") as f:
                f.write(result_response.content)
            print("Image generation complete!")
            break
        else:
            print("Still processing...")
            import time
            time.sleep(5)  # Wait 5 seconds before checking again
```

## Acknowledgments

This API is based on the [Diffusion Self-Distillation](https://github.com/primecai/diffusion-self-distillation) model by Shengqu Cai et al. Please cite their work if you use this API in your research or applications:

```bibtex
@inproceedings{cai2024dsd,
    author={Cai, Shengqu and Chan, Eric Ryan and Zhang, Yunzhi and Guibas, Leonidas and Wu, Jiajun and Wetzstein, Gordon.},
    title={Diffusion Self-Distillation for Zero-Shot Customized Image Generation},
    booktitle={CVPR},
    year={2025}
}
```# dsd-api
