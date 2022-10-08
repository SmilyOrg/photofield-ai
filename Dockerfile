FROM python:3.10-slim

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.2.1

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY models/clip-*.onnx models/
COPY cliponnx cliponnx
COPY main.py ./

EXPOSE 8081
CMD ["python", "main.py"]
