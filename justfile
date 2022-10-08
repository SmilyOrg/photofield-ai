set dotenv-load := true

default:
  @just --list --list-heading $'photofield-ai\n'

run:
  poetry run python main.py

docker:
  poetry export --without gpu -f requirements.txt > requirements.txt
  docker build -t photofield-ai.fat .
  docker run -it --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    dslim/docker-slim \
    build \
    --tag photofield-ai \
    --include-shell \
    --exclude-pattern=/usr/local/lib/python3.10/site-packages/protobuf-3.20.1-py3.10-nspkg.pth \
    photofield-ai.fat

compile:
  python -m nuitka --follow-imports main.py

watch:
  uvicorn main:app --reload --port 8081
