import asyncio
from os import environ
import io
import base64

from PIL import Image
import numpy as np

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import uvicorn
from cliponnx.download import ensure_model

from cliponnx.models import VisualModel, TextualModel, get_available_providers

app = FastAPI()
host = environ.get("PHOTOFIELD_AI_HOST", default="0.0.0.0")
port = environ.get("PHOTOFIELD_AI_PORT", default="8081")
models_dir = environ.get("PHOTOFIELD_AI_MODELS_DIR", default="models/")
visual_path = environ.get("PHOTOFIELD_AI_VISUAL_MODEL", default="https://huggingface.co/mlunar/clip-variants/resolve/main/models/clip-vit-base-patch32-visual-float16.onnx")
textual_path = environ.get("PHOTOFIELD_AI_TEXTUAL_MODEL", default="https://huggingface.co/mlunar/clip-variants/resolve/main/models/clip-vit-base-patch32-textual-float16.onnx")
runtime = environ.get("PHOTOFIELD_AI_RUNTIME", default="all")
providers_env = environ.get("PHOTOFIELD_AI_PROVIDERS")

# Debug comparison against another model
visual_comp_path = None
textual_comp_path = None

visual = None
textual = None
visual_comp = None
textual_comp = None

input_size = 0
input_name = None
output_name = None

async def run_async(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)

@app.on_event("startup")
async def startup_event():
    dir = "models/"
    global providers, visual, textual, visual_comp, textual_comp

    visual_file_path = ensure_model(visual_path, models_dir)
    textual_file_path = ensure_model(textual_path, models_dir)

    available_providers = get_available_providers()
    available_providers_str = ", ".join(available_providers)
    print(f"Available providers: {available_providers_str}")

    if runtime == "cpu":
        providers = ["CPUExecutionProvider"]
    elif runtime == "all":
        if providers_env is None:
            providers = get_available_providers()
        else:
            providers = providers_env.split(",")
    else:
        raise ValueError(f"Unsupported runtime {runtime}, use 'cpu', 'all' or leave empty for defaults")

    providers_str = ", ".join(providers)
    print(f"Using providers: {providers_str}")
    visual, textual, visual_comp, textual_comp = await asyncio.gather(*[
        run_async(VisualModel, visual_file_path, providers),
        run_async(TextualModel, textual_file_path, providers),
        run_async(VisualModel, visual_comp_path, providers) if visual_comp_path is not None else asyncio.sleep(0),
        run_async(TextualModel, textual_comp_path, providers) if textual_comp_path is not None else asyncio.sleep(0),
    ])
    print(f"Listening on {host}:{port}")

def encode_embedding(emb):
    inv_norm = np.divide(1, np.linalg.norm(emb), dtype=np.float16)
    inv_norm_uint16 = int.from_bytes(inv_norm.tobytes(), "little")
    tensor_b64 = base64.b64encode(emb.astype(np.float16))
    return tensor_b64, inv_norm_uint16

def compare(a_path, a, b_path, b):
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"A: {a_path}")
    print(f"B: {b_path}")
    print(f"Similarity: {similarity}")

@app.post("/image-embeddings")
async def post_image_embeddings(request: Request):
    form = await request.form()
    images = []
    items = list(form.items())
    for _, file in items:
        img = Image.open(io.BytesIO(await file.read()))
        img_np = await run_async(visual.preprocess, img)
        images.append(img_np)

    if len(images) == 0:
        raise HTTPException(status_code=400, detail="No images provided")

    image_input = np.stack(images)

    response_images = []
    image_features = await run_async(visual.encode, image_input)
    
    if visual_comp is not None:
        image_features_comp = await run_async(visual_comp.encode, image_input)
        compare(visual.path, image_features[0], visual_comp.path, image_features_comp[0])

    for i in range(len(items)):
        field, file = items[i]
        tensor_b64, inv_norm_uint16 = encode_embedding(image_features[i])
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
    text = textual.tokenize(b.texts)
    response_texts = []
    text_features = textual.encode(text)
    if textual_comp is not None:
        text_features_comp = textual_comp.encode(textual_comp.tokenize(b.texts))
        for index, text in enumerate(b.texts):
            print(f"Text: {text}")
            compare(textual.path, text_features[index], textual_comp.path, text_features_comp[index])
    for index, text in enumerate(b.texts):
        tensor_b64, inv_norm_uint16 = encode_embedding(text_features[index])
        response_texts.append({
            "text": text,
            "embedding_f16_b64": tensor_b64,
            "embedding_inv_norm_f16_uint16": inv_norm_uint16,
        })
    return {
        "texts": response_texts
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host=host, port=port, log_level="warning")
