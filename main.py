from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import io
import sys
import uuid
import shutil
from PIL import Image
from dotenv import load_dotenv
import torch
from typing import Optional

# Add parent directory to path to import from the original repo
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'diffusion-self-distillation'))

# Import from the original DSD repo
from pipeline import FluxConditionalPipeline
from transformer import FluxTransformer2DConditionalModel
from recaption import enhance_prompt

# Load environment variables
load_dotenv()

app = FastAPI(title="Diffusion Self-Distillation API", 
              description="API for Diffusion Self-Distillation for Zero-Shot Customized Image Generation",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Mount static files directory for web UI
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables
MODEL_PATH = os.environ.get("MODEL_PATH", "models/transformer")
LORA_PATH = os.environ.get("LORA_PATH", "models/pytorch_lora_weights.safetensors")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")

# Initialize directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

pipe = None

def init_pipeline():
    """Initialize the model pipeline if not already initialized"""
    global pipe
    if pipe is None:
        try:
            print(f"Loading model from {MODEL_PATH} and LORA from {LORA_PATH}")
            transformer = FluxTransformer2DConditionalModel.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True
            )
            pipe = FluxConditionalPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                transformer=transformer,
                torch_dtype=torch.bfloat16
            )
            pipe.load_lora_weights(LORA_PATH)
            
            # Choose between CPU offload or full GPU
            if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 24 * 1024 * 1024 * 1024:
                pipe.to("cuda")
                print("Model loaded on CUDA")
            else:
                pipe.enable_model_cpu_offload()
                print("Model loaded with CPU offload")
                
            return True
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            return False
    return True

def process_image(
    image_path: str, 
    text: str, 
    use_gemini: bool, 
    guidance: float, 
    i_guidance: float, 
    t_guidance: float, 
    output_path: str
):
    """Process the given image with the model"""
    global pipe
    
    if not init_pipeline():
        raise RuntimeError("Failed to initialize model pipeline")
    
    # Open and process the image
    image = Image.open(image_path).convert("RGB")
    
    # Center-crop and resize image
    w, h = image.size
    min_size = min(w, h)
    image = image.crop(((w - min_size) // 2, 
                        (h - min_size) // 2, 
                        (w + min_size) // 2, 
                        (h + min_size) // 2))
    image = image.resize((512, 512))
    
    # Enhance prompt using Gemini if enabled
    if use_gemini:
        try:
            enhanced_text = enhance_prompt(image, text.strip().replace("\n", "").replace("\r", ""))
            print(f"Original prompt: {text}")
            print(f"Enhanced prompt: {enhanced_text}")
            text = enhanced_text
        except Exception as e:
            print(f"Warning: Failed to enhance prompt: {str(e)}")
            # Continue with the original prompt
    
    # Process with the model
    result = pipe(
        prompt=text.strip().replace("\n", "").replace("\r", ""),
        negative_prompt="",
        num_inference_steps=28,
        height=512,
        width=1024,
        guidance_scale=guidance,
        image=image,
        guidance_scale_real_i=i_guidance,
        guidance_scale_real_t=t_guidance,
        gemini_prompt=use_gemini,
    ).images[0]
    
    # Save the result
    result.save(output_path)
    return output_path

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI"""
    with open("static/index.html", "r") as f:
        return f.read()

@app.get("/api")
async def api_info():
    """API info endpoint - check if the API is running"""
    return {"message": "Diffusion Self-Distillation API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint to verify if the model is loaded and ready"""
    if init_pipeline():
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}

@app.post("/generate/")
async def generate_image(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    text: str = Form(...),
    use_gemini: bool = Form(True),
    guidance: float = Form(3.5),
    i_guidance: float = Form(1.0),
    t_guidance: float = Form(1.0)
):
    """
    Generate an image using the Diffusion Self-Distillation model.
    
    - **image**: The conditioning image file
    - **text**: The text prompt
    - **use_gemini**: Whether to enhance the prompt using Gemini (requires API key)
    - **guidance**: The guidance scale (default: 3.5)
    - **i_guidance**: Image guidance scale (default: 1.0)
    - **t_guidance**: Text guidance scale (default: 1.0)
    """
    
    # Generate a unique ID for this request
    request_id = str(uuid.uuid4())
    
    # Save the uploaded image
    image_path = os.path.join(UPLOAD_DIR, f"{request_id}_input.png")
    output_path = os.path.join(OUTPUT_DIR, f"{request_id}_output.png")
    
    try:
        # Save the uploaded file
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Process the image (non-blocking if background_tasks is used)
        if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 24 * 1024 * 1024 * 1024:
            # For high-memory GPUs, process immediately
            process_image(
                image_path=image_path,
                text=text,
                use_gemini=use_gemini,
                guidance=guidance,
                i_guidance=i_guidance,
                t_guidance=t_guidance,
                output_path=output_path
            )
            return FileResponse(output_path, media_type="image/png")
        else:
            # For lower-memory systems, process in background
            background_tasks.add_task(
                process_image,
                image_path=image_path,
                text=text,
                use_gemini=use_gemini,
                guidance=guidance,
                i_guidance=i_guidance,
                t_guidance=t_guidance,
                output_path=output_path
            )
            return JSONResponse(
                content={
                    "message": "Processing started",
                    "request_id": request_id,
                    "status": "processing"
                }
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/result/{request_id}")
async def get_result(request_id: str):
    """
    Get the result of a previously submitted generation request.
    
    - **request_id**: The ID returned from the /generate endpoint
    """
    output_path = os.path.join(OUTPUT_DIR, f"{request_id}_output.png")
    
    if os.path.exists(output_path):
        return FileResponse(output_path, media_type="image/png")
    else:
        # Check if the request is still processing
        input_path = os.path.join(UPLOAD_DIR, f"{request_id}_input.png")
        if os.path.exists(input_path):
            return JSONResponse(
                status_code=202,
                content={
                    "message": "Processing not complete",
                    "request_id": request_id,
                    "status": "processing"
                }
            )
        else:
            raise HTTPException(status_code=404, detail=f"No result found for request ID: {request_id}")

@app.on_event("startup")
async def startup_event():
    """Initialize the model when the API starts"""
    # We don't initialize the model here to avoid slowing down startup
    # It will be initialized on the first request
    pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)