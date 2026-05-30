import base64
import io
import json
import os
import socket
import subprocess
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ORIENTATION_TAG = 274
ROOT_DIR = Path(__file__).resolve().parent.parent
FACES_SAMPLE_PATH = ROOT_DIR / "faces.jpg"
TESTDATA_ROOT = ROOT_DIR / "testdata" / "exif-orientation-e2e"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_health(api_url: str, timeout_s: float = 180.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{api_url}/health", timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(1.0)
    raise TimeoutError(f"API did not become healthy within {timeout_s:.0f}s: {api_url}")


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def post_multipart_image(api_url: str, path: str, field_name: str, filename: str, data: bytes) -> dict:
    boundary = "----photofield-ai-orientation-boundary"
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode("utf-8"))
    body.write(f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8"))
    body.write(b"Content-Type: image/jpeg\r\n\r\n")
    body.write(data)
    body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        f"{api_url}{path}",
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120.0) as resp:
        return json.loads(resp.read())


def decode_f16_embedding(embedding_b64: str) -> np.ndarray:
    raw = base64.b64decode(embedding_b64)
    return np.frombuffer(raw, dtype=np.float16).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def image_to_jpeg_bytes(image: Image.Image, orientation: int | None = None) -> bytes:
    exif = Image.Exif()
    if orientation is not None:
        exif[ORIENTATION_TAG] = orientation

    buf = io.BytesIO()
    if orientation is None:
        image.save(buf, format="JPEG", quality=95)
    else:
        image.save(buf, format="JPEG", quality=95, exif=exif)
    return buf.getvalue()


def create_asymmetric_test_image(size: tuple[int, int] = (320, 200)) -> Image.Image:
    image = Image.new("RGB", size, color=(238, 242, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle([10, 10, 140, 190], fill=(220, 40, 40))
    draw.polygon([(220, 25), (300, 50), (235, 110)], fill=(30, 120, 230))
    draw.ellipse([165, 115, 310, 190], fill=(245, 180, 35))
    draw.rectangle([150, 20, 165, 190], fill=(30, 30, 30))
    draw.rectangle([170, 20, 185, 190], fill=(245, 245, 245))
    return image


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def center_distance(a: list[float], b: list[float]) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def greedy_match_faces_by_center(
    a_faces: list[dict],
    b_faces: list[dict],
) -> list[tuple[dict, dict, float]]:
    if len(a_faces) != len(b_faces):
        raise ValueError("face list lengths differ")

    remaining = set(range(len(b_faces)))
    matches: list[tuple[dict, dict, float]] = []

    ordered_a = sorted(a_faces, key=lambda f: (bbox_center(f["bbox"])[0], bbox_center(f["bbox"])[1]))
    for a_face in ordered_a:
        best_idx = None
        best_dist = float("inf")
        for idx in remaining:
            dist = center_distance(a_face["bbox"], b_faces[idx]["bbox"])
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is None:
            raise ValueError("failed to match all faces")
        matches.append((a_face, b_faces[best_idx], best_dist))
        remaining.remove(best_idx)

    return matches


def face_pair_metrics(
    source_faces: list[dict],
    target_faces: list[dict],
    *,
    require_equal_count: bool,
) -> dict:
    metrics: dict = {
        "source_count": len(source_faces),
        "target_count": len(target_faces),
    }

    if require_equal_count and len(source_faces) != len(target_faces):
        raise AssertionError(
            f"face counts differ: source={len(source_faces)} target={len(target_faces)}"
        )

    if len(source_faces) == 0 or len(target_faces) == 0:
        metrics["matches"] = []
        return metrics

    if len(source_faces) != len(target_faces):
        metrics["matches"] = []
        return metrics

    matches = greedy_match_faces_by_center(source_faces, target_faces)
    rows = []
    cosines = []
    for source_face, target_face, center_dist in matches:
        sb = source_face["bbox"]
        tb = target_face["bbox"]
        emb_a = decode_f16_embedding(source_face["embedding_f16_b64"])
        emb_b = decode_f16_embedding(target_face["embedding_f16_b64"])
        sim = cosine_similarity(emb_a, emb_b)
        cosines.append(sim)
        rows.append(
            {
                "source_bbox": [float(v) for v in sb],
                "target_bbox": [float(v) for v in tb],
                "center_distance": center_dist,
                "bbox_abs_deltas": [abs(float(sb[i]) - float(tb[i])) for i in range(4)],
                "cosine_similarity": sim,
            }
        )

    metrics["matches"] = rows
    metrics["mean_cosine"] = float(np.mean(cosines))
    return metrics


class ExifOrientationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.port = find_free_port()
        cls.api_url = f"http://127.0.0.1:{cls.port}"
        cls.run_id = time.strftime("%Y%m%d-%H%M%S")
        cls.artifact_dir = TESTDATA_ROOT / cls.run_id
        cls.artifact_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["PHOTOFIELD_AI_HOST"] = "127.0.0.1"
        env["PHOTOFIELD_AI_PORT"] = str(cls.port)
        env["PHOTOFIELD_AI_RUNTIME"] = "cpu"

        cls.server_log = cls.artifact_dir / "server.log"
        cls._server_log_fp = cls.server_log.open("w", encoding="utf-8")
        cls.server_proc = subprocess.Popen(
            ["uv", "run", "python", "main.py"],
            cwd=str(ROOT_DIR),
            env=env,
            stdout=cls._server_log_fp,
            stderr=subprocess.STDOUT,
        )

        try:
            wait_for_health(cls.api_url)
        except Exception:
            cls.server_proc.terminate()
            cls.server_proc.wait(timeout=20)
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "server_proc") and cls.server_proc.poll() is None:
            cls.server_proc.terminate()
            try:
                cls.server_proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                cls.server_proc.kill()
                cls.server_proc.wait(timeout=20)
        if hasattr(cls, "_server_log_fp") and not cls._server_log_fp.closed:
            cls._server_log_fp.close()

    def test_image_embeddings_rot90_vs_rot90_exif8_behavior(self):
        original = create_asymmetric_test_image()
        rot90 = original.transpose(Image.Transpose.ROTATE_270)
        rot90_exif8 = rot90.copy()

        emb_original_payload = post_multipart_image(
            self.api_url,
            "/image-embeddings",
            "image",
            "original.jpg",
            image_to_jpeg_bytes(original),
        )
        emb_rot90_payload = post_multipart_image(
            self.api_url,
            "/image-embeddings",
            "image",
            "rot90.jpg",
            image_to_jpeg_bytes(rot90),
        )
        emb_rot90_exif8_payload = post_multipart_image(
            self.api_url,
            "/image-embeddings",
            "image",
            "rot90_exif8.jpg",
            image_to_jpeg_bytes(rot90_exif8, orientation=8),
        )

        emb_original = decode_f16_embedding(emb_original_payload["images"][0]["embedding_f16_b64"])
        emb_rot90 = decode_f16_embedding(emb_rot90_payload["images"][0]["embedding_f16_b64"])
        emb_rot90_exif8 = decode_f16_embedding(emb_rot90_exif8_payload["images"][0]["embedding_f16_b64"])

        sim_orig_vs_rot90 = cosine_similarity(emb_original, emb_rot90)
        sim_orig_vs_rot90_exif8 = cosine_similarity(emb_original, emb_rot90_exif8)
        sim_rot90_vs_rot90_exif8 = cosine_similarity(emb_rot90, emb_rot90_exif8)

        write_json(
            self.artifact_dir / "image_metrics.json",
            {
                "api_url": self.api_url,
                "run_id": self.run_id,
                "sim_orig_vs_rot90": sim_orig_vs_rot90,
                "sim_orig_vs_rot90_exif8": sim_orig_vs_rot90_exif8,
                "sim_rot90_vs_rot90_exif8": sim_rot90_vs_rot90_exif8,
                "thresholds": {
                    "sim_orig_vs_rot90_max": 0.97,
                    "sim_orig_vs_rot90_exif8_min": 0.99,
                    "delta_exif8_minus_rot90_min": 0.03,
                },
            },
        )

        # From the original image perspective, raw 90-degree rotation should
        # reduce similarity, while EXIF-corrected rotation should match closely.
        self.assertLessEqual(sim_orig_vs_rot90, 0.97)
        self.assertGreaterEqual(sim_orig_vs_rot90_exif8, 0.99)
        self.assertGreaterEqual(sim_orig_vs_rot90_exif8 - sim_orig_vs_rot90, 0.03)

    def test_faces_match_count_bbox_and_embedding_similarity_with_exif_rotation(self):
        if not FACES_SAMPLE_PATH.exists():
            self.skipTest(f"missing sample image: {FACES_SAMPLE_PATH}")

        base = Image.open(FACES_SAMPLE_PATH).convert("RGB")
        rot90 = base.transpose(Image.Transpose.ROTATE_270)

        original_payload = post_multipart_image(
            self.api_url,
            "/faces",
            "image0",
            "faces_original.jpg",
            image_to_jpeg_bytes(base),
        )
        rot90_payload = post_multipart_image(
            self.api_url,
            "/faces",
            "image0",
            "faces_rot90.jpg",
            image_to_jpeg_bytes(rot90),
        )
        rot90_exif8_payload = post_multipart_image(
            self.api_url,
            "/faces",
            "image0",
            "faces_rot90_exif8.jpg",
            image_to_jpeg_bytes(rot90, orientation=8),
        )

        original_faces = original_payload["images"][0]["faces"]
        rot90_faces = rot90_payload["images"][0]["faces"]
        exif_faces = rot90_exif8_payload["images"][0]["faces"]

        write_json(self.artifact_dir / "faces_original_response.json", original_payload)
        write_json(self.artifact_dir / "faces_rot90_response.json", rot90_payload)
        write_json(self.artifact_dir / "faces_rot90_exif8_response.json", rot90_exif8_payload)

        self.assertGreater(len(original_faces), 0)
        self.assertEqual(len(original_faces), len(exif_faces))

        bbox_tol_px = 6.0
        center_tol_px = 8.0
        min_face_cosine = 0.98

        orig_vs_exif = face_pair_metrics(
            original_faces,
            exif_faces,
            require_equal_count=True,
        )

        match_rows = orig_vs_exif["matches"]
        for row in match_rows:
            center_dist = float(row["center_distance"])
            self.assertLessEqual(center_dist, center_tol_px)
            for delta in row["bbox_abs_deltas"]:
                self.assertLessEqual(float(delta), bbox_tol_px)
            sim = float(row["cosine_similarity"])
            self.assertGreaterEqual(sim, min_face_cosine)

        mean_cosine = float(orig_vs_exif["mean_cosine"])

        orig_vs_rot90 = face_pair_metrics(
            original_faces,
            rot90_faces,
            require_equal_count=False,
        )

        # If both sets have comparable face detections, EXIF-corrected should
        # align better with original than raw rotated input.
        if (
            len(rot90_faces) == len(original_faces)
            and len(rot90_faces) > 0
            and "mean_cosine" in orig_vs_rot90
        ):
            self.assertGreater(
                mean_cosine,
                float(orig_vs_rot90["mean_cosine"]) + 0.02,
            )

        write_json(
            self.artifact_dir / "face_metrics.json",
            {
                "api_url": self.api_url,
                "run_id": self.run_id,
                "face_count": len(original_faces),
                "rot90_face_count": len(rot90_faces),
                "rot90_exif8_face_count": len(exif_faces),
                "thresholds": {
                    "bbox_tol_px": bbox_tol_px,
                    "center_tol_px": center_tol_px,
                    "min_face_cosine": min_face_cosine,
                    "mean_face_cosine_min": 0.99,
                },
                "orig_vs_rot90": orig_vs_rot90,
                "orig_vs_rot90_exif8": {
                    "mean_cosine": mean_cosine,
                    "matches": match_rows,
                },
            },
        )

        self.assertGreaterEqual(mean_cosine, 0.99)


if __name__ == "__main__":
    unittest.main()
