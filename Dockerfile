FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir huggingface_hub tqdm

# Copy the code files
COPY . .

# Copy the parent directory
RUN mkdir -p /app/diffusion-self-distillation
COPY ../diffusion-self-distillation/*.py /app/diffusion-self-distillation/

# Create directories for outputs and uploads
RUN mkdir -p outputs uploads models

# Download the model files (commented out by default, you'll need to run this or provide the models)
# RUN python download_model.py --use_hf

# Expose the port
EXPOSE 8000

# Set the environment variables
ENV MODEL_PATH=/app/models/transformer
ENV LORA_PATH=/app/models/pytorch_lora_weights.safetensors
ENV OUTPUT_DIR=/app/outputs
ENV UPLOAD_DIR=/app/uploads

# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]