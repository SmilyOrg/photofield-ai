# Get dependency requirements
FROM python:3.10-slim AS reqs
ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.2.1

RUN pip install "poetry==$POETRY_VERSION"
WORKDIR /app
COPY poetry.lock pyproject.toml ./
RUN poetry export --without gpu -f requirements.txt > requirements.txt

# Runtime
FROM python:3.10-slim
ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100

WORKDIR /app
COPY --from=reqs /app/requirements.txt ./
RUN pip install -r requirements.txt

COPY models/clip-*.onnx models/
COPY cliponnx cliponnx
COPY main.py ./

EXPOSE 8081
CMD ["python", "main.py"]
