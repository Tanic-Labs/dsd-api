#!/usr/bin/env python3
import os
import argparse
import requests
from tqdm import tqdm
import json
import shutil

def download_file(url, local_filename):
    """Download a file with progress bar"""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(local_filename, 'wb') as f, tqdm(
                desc=local_filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return local_filename

def main():
    parser = argparse.ArgumentParser(description="Download DSD Model")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save the model")
    parser.add_argument("--use_hf", action="store_true", help="Download from HuggingFace instead of Google Drive")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.use_hf:
        # Download from HuggingFace
        print("Downloading from HuggingFace (primecai/dsd_model)...")
        from huggingface_hub import hf_hub_download, snapshot_download
        
        # Download the entire repository snapshot
        model_path = snapshot_download(
            repo_id="primecai/dsd_model",
            local_dir=args.output_dir
        )
        print(f"Model downloaded to {model_path}")
        
    else:
        # Download from Google Drive (as described in README)
        print("Downloading from Google Drive...")
        
        # URLs for the model files (from the README)
        gdrive_url_base = "https://drive.google.com/uc?export=download&id="
        
        # File IDs from the README's Google Drive link
        model_files = {
            "transformers/config.json": "1VStt7J2whm5RRloa4NK1hGTHuS9WiTfO",
            "transformers/diffusion_pytorch_model.safetensors": "1VStt7J2whm5RRloa4NK1hGTHuS9WiTfO",
            "pytorch_lora_weights.safetensors": "1VStt7J2whm5RRloa4NK1hGTHuS9WiTfO"
        }
        
        # Create the transformers directory
        os.makedirs(os.path.join(args.output_dir, "transformer"), exist_ok=True)
        
        for file_path, file_id in model_files.items():
            output_path = os.path.join(args.output_dir, file_path.replace("transformers", "transformer"))
            url = f"{gdrive_url_base}{file_id}"
            print(f"Downloading {file_path} to {output_path}...")
            download_file(url, output_path)
    
    print("Model download complete!")

if __name__ == "__main__":
    main()