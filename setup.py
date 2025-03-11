#!/usr/bin/env python3
import os
import shutil
import argparse
import subprocess

def create_directories():
    """Create necessary directories"""
    dirs = ["models", "outputs", "uploads", "static"]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            print(f"Created directory: {d}")

def setup_environment():
    """Set up the environment file if it doesn't exist"""
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            shutil.copy(".env.example", ".env")
            print("Created .env file from .env.example")
        else:
            # Create a basic .env file
            with open(".env", "w") as f:
                f.write("# Model paths\n")
                f.write("MODEL_PATH=models/transformer\n")
                f.write("LORA_PATH=models/pytorch_lora_weights.safetensors\n\n")
                f.write("# Storage directories\n")
                f.write("OUTPUT_DIR=outputs\n")
                f.write("UPLOAD_DIR=uploads\n\n")
                f.write("# Google Gemini API key (required for prompt enhancement)\n")
                f.write("GOOGLE_API_KEY=your_google_api_key\n")
            print("Created default .env file")
    else:
        print(".env file already exists")

def download_model():
    """Download the model files if they don't exist"""
    if not os.path.exists("models/transformer") or not os.path.exists("models/pytorch_lora_weights.safetensors"):
        print("Downloading model files...")
        subprocess.run(["python", "download_model.py", "--use_hf"])
    else:
        print("Model files already exist")

def install_dependencies():
    """Install required Python dependencies"""
    print("Installing Python dependencies...")
    subprocess.run(["pip", "install", "-r", "requirements.txt"])

def main():
    parser = argparse.ArgumentParser(description="Set up the Diffusion Self-Distillation API")
    parser.add_argument("--skip-deps", action="store_true", help="Skip installing dependencies")
    parser.add_argument("--skip-model", action="store_true", help="Skip downloading model files")
    args = parser.parse_args()
    
    print("Setting up Diffusion Self-Distillation API...")
    
    # Create directories
    create_directories()
    
    # Set up environment file
    setup_environment()
    
    # Install dependencies
    if not args.skip_deps:
        install_dependencies()
    
    # Download model
    if not args.skip_model:
        download_model()
    
    print("\nSetup complete! You can now run the API with:")
    print("  uvicorn main:app --host 0.0.0.0 --port 8000")
    print("\nOr use Docker:")
    print("  ./run_docker.sh")

if __name__ == "__main__":
    main()