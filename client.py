#!/usr/bin/env python3
import argparse
import requests
import time
import os
from PIL import Image

def generate_image(
    server_url, 
    image_path, 
    prompt, 
    output_path, 
    guidance=3.5, 
    i_guidance=1.0, 
    t_guidance=1.0, 
    use_gemini=True
):
    """Generate an image using the Diffusion Self-Distillation API"""
    
    url = f"{server_url}/generate/"
    
    # Prepare the files and data
    files = {"image": open(image_path, "rb")}
    data = {
        "text": prompt,
        "use_gemini": str(use_gemini).lower(),
        "guidance": str(guidance),
        "i_guidance": str(i_guidance),
        "t_guidance": str(t_guidance)
    }
    
    print(f"Sending request to {url}...")
    print(f"Prompt: {prompt}")
    
    # Send the request
    response = requests.post(url, files=files, data=data)
    
    # Check if the response is an image
    if response.headers.get("content-type") == "image/png":
        print("Received image directly!")
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Image saved to {output_path}")
        return True
    
    # If the response is JSON (async processing), get the request ID
    elif response.headers.get("content-type") == "application/json":
        result = response.json()
        request_id = result.get("request_id")
        
        if not request_id:
            print(f"Error: {result}")
            return False
        
        print(f"Processing started with request ID: {request_id}")
        
        # Poll the result endpoint until the image is ready
        while True:
            print("Checking if processing is complete...")
            result_response = requests.get(f"{server_url}/result/{request_id}")
            
            if result_response.headers.get("content-type") == "image/png":
                with open(output_path, "wb") as f:
                    f.write(result_response.content)
                print(f"Image generation complete! Saved to {output_path}")
                
                # Display the image
                try:
                    img = Image.open(output_path)
                    img.show()
                except:
                    pass
                
                return True
            elif result_response.status_code == 202:
                print("Still processing...")
                time.sleep(5)  # Wait 5 seconds before checking again
            else:
                print(f"Error: {result_response.text}")
                return False
    
    else:
        print(f"Unexpected response: {response.text}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Client for Diffusion Self-Distillation API")
    parser.add_argument("--server", type=str, default="http://localhost:8000", help="Server URL")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--i_guidance", type=float, default=1.0, help="Image guidance scale")
    parser.add_argument("--t_guidance", type=float, default=1.0, help="Text guidance scale")
    parser.add_argument("--no_gemini", action="store_true", help="Disable Gemini prompt enhancement")
    
    args = parser.parse_args()
    
    # Verify the input image exists
    if not os.path.exists(args.image):
        print(f"Error: Input image {args.image} does not exist")
        return
    
    # Generate the image
    generate_image(
        server_url=args.server,
        image_path=args.image,
        prompt=args.prompt,
        output_path=args.output,
        guidance=args.guidance,
        i_guidance=args.i_guidance,
        t_guidance=args.t_guidance,
        use_gemini=not args.no_gemini
    )

if __name__ == "__main__":
    main()