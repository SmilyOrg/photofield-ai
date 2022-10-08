import io
import base64

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import uvicorn

import torch
import clip
from PIL import Image
import numpy as np

app = FastAPI()

model: torch.nn.Module = None
preprocess = None
device = None

@app.on_event("startup")
async def startup_event():
    global model, preprocess, device
    model_name = "ViT-B/32"
    print(f"Initializing CLIP")
    print(f"Model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Torch device: {device}")
    model, preprocess = clip.load(model_name, device=device)
    print("Done")

def tensor_to_embedding(tensor):
    inv_norm = 1/torch.linalg.vector_norm(tensor)
    inv_norm_uint16 = int.from_bytes(inv_norm.cpu().numpy().tobytes(), "little")

    tensor_np = tensor.cpu().numpy()
    print(tensor_np)
    tensor_b64 = base64.b64encode(tensor_np)
    return tensor_b64, inv_norm_uint16

@app.post("/image-embeddings")
async def post_image_embeddings(request: Request):
    form = await request.form()
    images = []
    items = list(form.items())
    for _, file in items:
        img = Image.open(io.BytesIO(await file.read()))
        images.append(preprocess(img))

    if len(images) == 0:
        raise HTTPException(status_code=400, detail="No images provided")

    image_input = torch.tensor(np.stack(images)).to(device)

    response_images = []
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        for i in range(len(items)):
            field, file = items[i]
            tensor_b64, inv_norm_uint16 = tensor_to_embedding(image_features[i])
            response_images.append({
                "field": field,
                "filename": file.filename,
                "embedding_f16_b64": tensor_b64,
                "embedding_inv_norm_f16_uint16": inv_norm_uint16,
            })
    return {
        "images": response_images
    }
    

class TextEmbeddings(BaseModel):
    texts: list[str]

@app.post("/text-embeddings")
async def post_text_embeddings(b: TextEmbeddings):
    text = clip.tokenize(b.texts).to(device)
    response_texts = []
    with torch.no_grad():
        text_features = model.encode_text(text)
        for index, text in enumerate(b.texts):
            tensor_b64, inv_norm_uint16 = tensor_to_embedding(text_features[index])
            response_texts.append({
                "text": text,
                "embedding_f16_b64": tensor_b64,
                "embedding_inv_norm_f16_uint16": inv_norm_uint16,
            })
    return {
        "texts": response_texts
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8081)