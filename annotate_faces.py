#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import uuid
from pathlib import Path
from urllib import error, request

import cv2


def encode_multipart_formdata(field_name: str, file_path: Path) -> tuple[bytes, str]:
    boundary = f"----photofield-ai-{uuid.uuid4().hex}"
    filename = file_path.name
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    file_bytes = file_path.read_bytes()

    lines = [
        f"--{boundary}".encode("utf-8"),
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"'.encode("utf-8"),
        f"Content-Type: {content_type}".encode("utf-8"),
        b"",
        file_bytes,
        f"--{boundary}--".encode("utf-8"),
        b"",
    ]
    body = b"\r\n".join(lines)
    return body, boundary


def post_faces(api_url: str, image_path: Path, field_name: str, timeout: float) -> dict:
    body, boundary = encode_multipart_formdata(field_name=field_name, file_path=image_path)
    req = request.Request(
        url=f"{api_url.rstrip('/')}/faces",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"/faces request failed with HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not connect to API at {api_url}: {exc.reason}") from exc

    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError("API did not return valid JSON") from exc


def draw_face_annotations(image_path: Path, output_path: Path, response: dict) -> int:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    images = response.get("images")
    if not isinstance(images, list) or not images:
        raise RuntimeError("Unexpected response shape: missing images[]")

    faces = images[0].get("faces", [])

    for index, face in enumerate(faces, start=1):
        bbox = face.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        conf = float(face.get("confidence", 0.0))
        inv_norm = int(face.get("embedding_inv_norm_f16_uint16", 0))
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)

        cv2.rectangle(image, (x1, y1), (x2, y2), (40, 215, 255), 2)

        label_lines = [
            f"face {index}",
            f"con: {conf:.3f}",
            f"pos: {x1},{y1}",
            f"dim: {width}x{height}",
            f"inv_norm: {inv_norm}",
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        line_gap = 4

        metrics = [cv2.getTextSize(line, font, font_scale, thickness) for line in label_lines]
        max_text_w = max((size[0] for size, _ in metrics), default=0)
        text_heights = [size[1] for size, _ in metrics]
        baselines = [bl for _, bl in metrics]

        box_w = max_text_w + 8
        box_h = sum(text_heights) + sum(baselines) + line_gap * (len(label_lines) - 1) + 8

        tx = max(0, min(x1, image.shape[1] - box_w - 1))
        ty = y1 - box_h - 6
        if ty < 0:
            ty = min(image.shape[0] - box_h - 1, y1 + 6)

        cv2.rectangle(
            image,
            (tx, ty),
            (tx + box_w, ty + box_h),
            (20, 20, 20),
            -1,
        )

        cursor_y = ty + 4
        for line, ((_, text_h), baseline) in zip(label_lines, metrics):
            text_y = cursor_y + text_h
            cv2.putText(image, line, (tx + 4, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            cursor_y = text_y + baseline + line_gap

        landmarks = face.get("landmarks", [])
        if isinstance(landmarks, list):
            for point in landmarks:
                if isinstance(point, list) and len(point) == 2:
                    px, py = int(round(point[0])), int(round(point[1]))
                    cv2.circle(image, (px, py), 2, (80, 255, 120), -1)

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write output image: {output_path}")

    return len(faces)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call /faces API and render face boxes + metadata on an output image.",
    )
    parser.add_argument("input", type=Path, help="Path to input image")
    parser.add_argument("output", type=Path, help="Path to output image")
    parser.add_argument("--api", default="http://localhost:8081", help="API base URL")
    parser.add_argument("--field", default="image0", help="Multipart form field name")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input image not found: {args.input}")

    response = post_faces(api_url=args.api, image_path=args.input, field_name=args.field, timeout=args.timeout)
    face_count = draw_face_annotations(image_path=args.input, output_path=args.output, response=response)
    print(f"Wrote {args.output} with {face_count} annotated face(s)")


if __name__ == "__main__":
    main()