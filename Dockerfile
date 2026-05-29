# Runtime
FROM python:3.13-slim
ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100

# uv is mounted at build time only (no persistent layer)

ARG VERSION=dev
ENV PHOTOFIELD_AI_VERSION=$VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies, patch uniface to eliminate scipy+scikit-image, then strip unused packages
RUN --mount=type=bind,from=ghcr.io/astral-sh/uv:latest,source=/uv,target=/bin/uv \
  uv sync --frozen --no-dev --no-install-project --no-cache --python $(which python3) \
  # uniface requires opencv-python but headless is sufficient; force-reinstall headless so
  # cv2.abi3.so links to headless libs (no Qt/X11) rather than the full opencv ones
  && uv pip install --no-cache --no-deps --force-reinstall opencv-python-headless \
  && SP=/app/.venv/lib/python3.13/site-packages \
  # Remove full opencv libs now that headless cv2.so is in place
  && rm -rf $SP/opencv_python.libs $SP/opencv_python-*.dist-info \
  # Remove Qt highgui module (not usable headless)
  && rm -rf $SP/cv2/qt \
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
  # Remove onnxruntime dev/optimization tools (not needed at inference time)
  && rm -rf $SP/onnxruntime/transformers $SP/onnxruntime/quantization \
             $SP/onnxruntime/tools $SP/onnxruntime/ThirdPartyNotices.txt \
  # Remove system Python extras not needed at runtime
  && rm -rf /usr/local/lib/python3.13/idlelib \
            /usr/local/lib/python3.13/tkinter \
            /usr/local/lib/python3.13/ensurepip \
            /usr/local/lib/python3.13/unittest \
            /usr/local/lib/python3.13/pydoc_data \
            /usr/local/lib/python3.13/turtle.py \
            /usr/local/lib/python3.13/turtledemo \
            /usr/local/lib/python3.13/site-packages/pip \
  # Remove CJK legacy-encoding codecs (Shift-JIS, GB2312, EUC-KR, Big5…)
  # UTF-8 CJK is handled natively; these are only needed for old byte-encoded text
  && rm -f /usr/local/lib/python3.13/lib-dynload/_codecs_jp.* \
           /usr/local/lib/python3.13/lib-dynload/_codecs_hk.* \
           /usr/local/lib/python3.13/lib-dynload/_codecs_cn.* \
           /usr/local/lib/python3.13/lib-dynload/_codecs_kr.* \
           /usr/local/lib/python3.13/lib-dynload/_codecs_tw.* \
           /usr/local/lib/python3.13/lib-dynload/_curses*.* \
           /usr/local/lib/python3.13/lib-dynload/_testcapi.* \
           /usr/local/lib/python3.13/lib-dynload/_testlimitedcapi.* \
           /usr/local/lib/python3.13/lib-dynload/_testclinic*.* \
  # Remove system packages no longer needed after apt install
  && rm -rf /usr/lib/x86_64-linux-gnu/perl-base \
            /usr/lib/x86_64-linux-gnu/libapt-pkg* \
            /usr/lib/x86_64-linux-gnu/libdb-5.3* \
  # Remove locale charset converters
  && rm -rf /usr/lib/x86_64-linux-gnu/gconv \


# Inject numpy-only SimilarityTransform stub (replaces skimage.transform dependency)
COPY patches/uniface_similarity.py /app/.venv/lib/python3.13/site-packages/uniface/_similarity.py

COPY models/clip-*.onnx models/
COPY cliponnx cliponnx
COPY main.py ./

EXPOSE 8081
CMD ["/app/.venv/bin/python", "main.py"]
