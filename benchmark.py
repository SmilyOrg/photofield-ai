#!/usr/bin/env python3
"""
Simple benchmark script for Photofield AI performance testing.
Tests both text and image embedding performance.
"""

import json
import time
import io
import urllib.request
import cv2
import numpy as np
from pathlib import Path
from statistics import mean, stdev

from PIL import Image
from uniface.constants import EdgeFaceWeights
from uniface.detection import RetinaFace
from uniface.recognition import EdgeFace


API_URL = "http://localhost:8081"
WARMUP_REQUESTS = 5
BENCHMARK_REQUESTS = 50
FACE_BENCHMARK_RUNS = 50
FACE_SAMPLE_PATH = Path(__file__).with_name("faces.jpg")


def create_test_image(size=(224, 224)):
    """Create a simple test image in memory."""
    img = Image.new("RGB", size, color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return buf


def load_face_sample() -> np.ndarray:
    """Load the local face benchmark sample as an OpenCV image."""
    image = cv2.imread(str(FACE_SAMPLE_PATH))
    if image is None:
        raise FileNotFoundError(f"could not load face sample: {FACE_SAMPLE_PATH}")
    return image


def post_json(url: str, data: dict) -> dict:
    """POST JSON data to a URL."""
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read())


def post_multipart(url: str, files: dict) -> dict:
    """POST multipart form data to a URL."""
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    
    body = io.BytesIO()
    for name, (filename, data, content_type) in files.items():
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode())
        body.write(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.write(data.read())
        body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())
    
    req = urllib.request.Request(
        url,
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read())


def benchmark_text_embeddings():
    """Benchmark text embedding performance."""
    test_texts = [
        "a photo of a cat",
        "sunset over mountains",
        "city skyline at night",
        "person walking in park",
        "red sports car",
    ]
    
    print("Warming up text embeddings...")
    for _ in range(WARMUP_REQUESTS):
        post_json(f"{API_URL}/text-embeddings", {"texts": test_texts})
    
    print(f"Running {BENCHMARK_REQUESTS} text embedding requests...")
    times = []
    for i in range(BENCHMARK_REQUESTS):
        start = time.perf_counter()
        try:
            post_json(f"{API_URL}/text-embeddings", {"texts": test_texts})
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    return times


def benchmark_image_embeddings():
    """Benchmark image embedding performance."""
    print("Warming up image embeddings...")
    for _ in range(WARMUP_REQUESTS):
        img_data = create_test_image()
        post_multipart(
            f"{API_URL}/image-embeddings",
            {"image": ("test.jpg", img_data, "image/jpeg")},
        )
    
    print(f"Running {BENCHMARK_REQUESTS} image embedding requests...")
    times = []
    for i in range(BENCHMARK_REQUESTS):
        img_data = create_test_image()
        start = time.perf_counter()
        try:
            post_multipart(
                f"{API_URL}/image-embeddings",
                {"image": ("test.jpg", img_data, "image/jpeg")},
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    return times


def print_stats(name: str, times: list[float]):
    """Print statistics for a benchmark."""
    if not times:
        print(f"\n{name}: No data")
        return
    
    avg_time = mean(times)
    std_time = stdev(times) if len(times) > 1 else 0
    min_time = min(times)
    max_time = max(times)
    throughput = 1.0 / avg_time
    
    print(f"\n{name}:")
    print(f"  Requests:    {len(times)}")
    print(f"  Average:     {avg_time*1000:.2f} ms")
    print(f"  Std Dev:     {std_time*1000:.2f} ms")
    print(f"  Min:         {min_time*1000:.2f} ms")
    print(f"  Max:         {max_time*1000:.2f} ms")
    print(f"  Throughput:  {throughput:.2f} req/sec")


def benchmark_face_recognizers():
    """Benchmark EdgeFace XXS face recognition using the same detected face landmarks."""
    image = load_face_sample()

    print("Preparing face benchmark sample...")
    detector = RetinaFace()
    faces = detector.detect(image)
    if not faces:
        print("No faces detected in face benchmark sample")
        return []

    landmarks = faces[0].landmarks
    recognizer = EdgeFace(model_name=EdgeFaceWeights.XXS)

    print("Warming up EdgeFace XXS...")
    for _ in range(WARMUP_REQUESTS):
        recognizer.get_normalized_embedding(image, landmarks)

    print(f"Running {FACE_BENCHMARK_RUNS} EdgeFace XXS embedding calls...")
    times = []
    embedding_shape = None
    for _ in range(FACE_BENCHMARK_RUNS):
        start = time.perf_counter()
        embedding = recognizer.get_normalized_embedding(image, landmarks)
        elapsed = time.perf_counter() - start
        embedding_shape = embedding.shape
        times.append(elapsed)

    return [("EdgeFace XXS", embedding_shape, times)]


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("Photofield AI Benchmark")
    print("=" * 60)
    
    # Benchmark text embeddings
    text_times = benchmark_text_embeddings()
    
    # Benchmark image embeddings
    image_times = benchmark_image_embeddings()

    # Benchmark face recognizer directly
    face_results = benchmark_face_recognizers()
    
    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    print_stats("Text Embeddings (5 texts per request)", text_times)
    print_stats("Image Embeddings (224x224 JPEG)", image_times)
    for name, embedding_shape, times in face_results:
        print_stats(f"{name} Face Embeddings {embedding_shape}", times)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
