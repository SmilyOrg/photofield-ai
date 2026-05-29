# Runtime
FROM python:3.13-slim
ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG VERSION=dev
ENV PHOTOFIELD_AI_VERSION=$VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies, patch uniface to eliminate scipy+scikit-image, then strip unused packages
RUN uv sync --frozen --no-dev --no-install-project --no-cache --python $(which python3) \
  && SP=/app/.venv/lib/python3.13/site-packages \
  # Patch uniface to use our numpy-only SimilarityTransform instead of skimage
  && sed -i 's/from skimage.transform import SimilarityTransform/from uniface._similarity import SimilarityTransform/' $SP/uniface/face_utils.py \
  # Remove BYTETracker from uniface exports (it pulls in scipy via kalman filter)
  && sed -i '/from .tracking import BYTETracker/d' $SP/uniface/__init__.py \
  && sed -i "/'BYTETracker',/d" $SP/uniface/__init__.py \
  # Remove heavy packages not reachable from our import chain
  && rm -rf $SP/scipy $SP/scipy.libs $SP/skimage \
             $SP/sympy $SP/mpmath \
             $SP/networkx $SP/imageio $SP/tifffile \
  # Remove test suites
  && find $SP -type d \( -name 'tests' -o -name 'test' \) -exec rm -rf {} + 2>/dev/null; true \
  # Remove cv2 Haar cascades (ONNX-based detection is used instead)
  && rm -rf $SP/cv2/data/ \
  # Remove system Python extras not needed at runtime
  && rm -rf /usr/local/lib/python3.13/idlelib \
            /usr/local/lib/python3.13/tkinter \
            /usr/local/lib/python3.13/ensurepip \
            /usr/local/lib/python3.13/site-packages/pip \
  # Remove locale charset converters
  && rm -rf /usr/lib/x86_64-linux-gnu/gconv \
  # Remove uv after use (no longer needed at runtime)
  && rm -f /bin/uv /bin/uvx

# Inject numpy-only SimilarityTransform stub (replaces skimage.transform dependency)
COPY patches/uniface_similarity.py /app/.venv/lib/python3.13/site-packages/uniface/_similarity.py

COPY models/clip-*.onnx models/
COPY cliponnx cliponnx
COPY main.py ./

EXPOSE 8081
CMD ["/app/.venv/bin/python", "main.py"]
