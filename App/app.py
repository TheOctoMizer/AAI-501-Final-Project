import base64
import asyncio
import json
import os
from io import BytesIO
from PIL import Image
import torch
from get_model import load_det_text_model, load_det_image_model
from get_infer import get_device, predict_with_model, predict_image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Union

app = FastAPI()
templates = Jinja2Templates(directory="templates")
text_model = None
image_model = None
text_tokenizer = None
device = "cpu"
idx_to_class = None

# Model loading function
async def load_text_model():
    global text_model, text_tokenizer, device
    text_tokenizer, text_model = await load_det_text_model(device=device)
    print("text_model loaded successfully")

async def load_image_model():
    global device, image_model
    image_model = await load_det_image_model(device=device)
    print("image_model loaded successfully")

async def get_idx_class():
    global idx_to_class
    with open("./models/class_to_idx.json", "r") as f:
        idx_to_class = json.load(f)
        idx_to_class = {v: k for k, v in idx_to_class.items()}
    print("idx_to_class loaded successfully")


# Get inference device
async def get_infer_device():
    global device
    device = await get_device()
    return device

# Data model for input validation
class InputData(BaseModel):
    type: str
    text: Union[str, None] = None
    image: Union[str, None] = None

class RequestModel(BaseModel):
    inputs: List[InputData]

# Render template route
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Process route
@app.post("/process")
async def process(request_data: RequestModel):
    global idx_to_class, image_model, text_model, text_tokenizer, device
    processed_inputs = []
    for input_data in request_data.inputs:
        if input_data.text:
            # Process text input
            paragraphs, flags = await predict_with_model(
                input_data.text,
                text_model.half(),
                text_tokenizer,
                model_type="torch",
                device=device
            )
            processed_inputs.append({
                "type": "text",
                "paragraphs": paragraphs,
                "flags": flags
            })
        elif input_data.image:
            # Process image input
            image_data = base64.b64decode(input_data.image)
            image = Image.open(BytesIO(image_data)).convert("RGB")
            prediction = await predict_image(
                model=image_model,
                image_path=image,
                device=device,
                idx_to_class=idx_to_class
            )
            print(prediction)
            # label = idx_to_class['AI'] if prediction.argmax().item() == 0 else idx_to_class['Human']
            processed_inputs.append({
                "type": "image",
                "label": prediction
            })

    return JSONResponse({"processed_inputs": processed_inputs})

# Run setup on startup
@app.on_event("startup")
async def startup_event():
    await get_infer_device()
    await load_text_model()
    await load_image_model()
    await get_idx_class()

