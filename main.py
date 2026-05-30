import asyncio
from os import environ
import io
import base64
import logging
import sys
from contextlib import asynccontextmanager

from PIL import Image, ImageOps
import numpy as np
import cv2

from fastapi import FastAPI, Request, HTTPException, Response
from starlette.requests import ClientDisconnect
from pydantic import BaseModel
import uvicorn
from cliponnx.download import ensure_model

from cliponnx.models import VisualModel, TextualModel, get_available_providers
from pathlib import Path
from typing import Any

from uniface.detection import RetinaFace
from uniface.recognition import EdgeFace
from uniface.constants import EdgeFaceWeights

logging.basicConfig(format="%(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

version = environ.get("PHOTOFIELD_AI_VERSION", "dev")

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

visual: VisualModel
textual: TextualModel
visual_comp: VisualModel | None = None
textual_comp: TextualModel | None = None
face_detector: RetinaFace
face_recognizer: EdgeFace

input_size = 0
input_name = None
output_name = None

async def run_async(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize models and providers
    global providers, visual, textual, visual_comp, textual_comp, face_detector, face_recognizer
    
    models_path = Path(models_dir)
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory does not exist: {models_dir}")
    if not models_path.is_dir():
        raise NotADirectoryError(f"Models path is not a directory: {models_dir}")
    
    log.info("photofield-ai %s", version)

    visual_file_path = ensure_model(visual_path, models_dir)
    textual_file_path = ensure_model(textual_path, models_dir)

    available_providers = get_available_providers()
    log.info("providers available %s", ", ".join(available_providers))

    if runtime == "cpu":
        providers = ["CPUExecutionProvider"]
    elif runtime == "all":
        if providers_env is None:
            providers = get_available_providers()
        else:
            providers = providers_env.split(",")
    else:
        raise ValueError(f"Unsupported runtime {runtime}, use 'cpu', 'all' or leave empty for defaults")

    log.info("providers %s", ", ".join(providers))
    log.info("models initializing")
    visual, textual, visual_comp, textual_comp, face_detector, face_recognizer = await asyncio.gather(*[
        run_async(VisualModel, visual_file_path, providers),
        run_async(TextualModel, textual_file_path, providers),
        run_async(VisualModel, visual_comp_path, providers) if visual_comp_path is not None else asyncio.sleep(0),
        run_async(TextualModel, textual_comp_path, providers) if textual_comp_path is not None else asyncio.sleep(0),
        run_async(RetinaFace),
        run_async(lambda: EdgeFace(model_name=EdgeFaceWeights.XXS, providers=providers)),
    ])
    log.info("face models retinaface detection + edgeface xxs recognition")
    log.info("")
    log.info("listening on %s:%s", host, port)
    
    yield
    
    # Shutdown: Clean up resources if needed
    # Currently no cleanup needed, but you could add it here

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok"}

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

@app.head("/image-embeddings")
async def head_image_embeddings():
    return Response()

@app.post("/image-embeddings")
async def post_image_embeddings(request: Request):
    form = await request.form()
    images = []
    items = list(form.items())
    for _, file in items:
        img = Image.open(io.BytesIO(await file.read()))
        img = ImageOps.exif_transpose(img)
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

@app.head("/text-embeddings")
async def head_text_embeddings():
    return Response()

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

@app.head("/faces")
async def head_faces():
    return Response()

@app.post("/faces")
async def post_faces(request: Request):
    try:
        form = await request.form()
    except ClientDisconnect:
        return Response(status_code=499)
    items = list(form.items())
    response_images = []
    
    for field, file in items:
        img_bytes = await file.read()
        # Decode image directly with OpenCV from bytes
        img_array = np.frombuffer(img_bytes, np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_cv is None:
            raise HTTPException(status_code=400, detail=f"could not decode image: {file.filename}")

        # Detect faces
        faces = await run_async(face_detector.detect, img_cv)
        
        # Convert face results to serializable format
        face_results = []
        for face in faces:
            # Get face recognition embedding
            embedding = await run_async(face_recognizer.get_normalized_embedding, img_cv, face.landmarks)
            tensor_b64, inv_norm_uint16 = encode_embedding(embedding)
            
            face_results.append({
                "bbox": face.bbox.tolist(),  # [x1, y1, x2, y2]
                "confidence": float(face.confidence),
                "landmarks": face.landmarks.tolist(),  # 5-point landmarks [[x1, y1], [x2, y2], ...]
                "embedding_f16_b64": tensor_b64,
                "embedding_inv_norm_f16_uint16": inv_norm_uint16,
            })
        
        response_images.append({
            "field": field,
            "filename": file.filename,
            "faces": face_results,
        })
    
    return {
        "images": response_images
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="photofield-ai")
    parser.add_argument("--version", "-V", action="store_true", help="print version and exit")
    parser.add_argument("--preload", action="store_true", help="initialize all models then exit (for Docker build caching)")
    args = parser.parse_args()
    if args.version:
        print(f"photofield-ai {version}")
        sys.exit(0)
    if args.preload:
        async def _preload():
            async with lifespan(app):
                pass
        asyncio.run(_preload())
        sys.exit(0)
    uvicorn.run("main:app", host=host, port=int(port), log_level="warning")
