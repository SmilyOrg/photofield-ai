<!-- HEADER -->
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="assets/android-chrome-192x192.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Photofield AI</h3>

  <p align="center">
    Experimental machine learning API supporting <a href="https://github.com/SmilyOrg/photofield">Photofield</a>.
    <br />
    <br />
    <a href="https://github.com/SmilyOrg/photofield-ai/issues">Report Bug</a>
    ·
    <a href="https://github.com/SmilyOrg/photofield-ai/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About</a>
      <ul>
        <li><a href="#features">Features</a></li>
        <li><a href="#limitations">Limitations</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#configuration">Configuration</a></li>
    <li><a href="#troubleshooting">Troubleshooting</a></li>
    <li><a href="#development">Development</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



## About

Photofield AI is a machine learning companion service for [Photofield], providing semantic image search capabilities through [OpenAI CLIP] embeddings. It's a separate REST API service designed to keep the main app lightweight while leveraging Python's AI ecosystem.

**Quick Start:** `docker run -it -p 8081:8081 ghcr.io/smilyorg/photofield-ai:latest`

### Features

* **Fast CLIP Embeddings** - Convert images and text to semantic vectors for similarity search
* **Face Detection** - Detect faces in images with bounding boxes, confidence scores, and 5-point landmarks using RetinaFace
* **High Performance** - ~20 req/sec (i7-5820K CPU), ~200 req/sec (GTX 1070 Ti GPU)
* **Multiple Models** - Support for various CLIP model sizes and quantization levels
* **Easy Integration** - Simple REST API with multipart image uploads
* **Docker Ready** - Pre-built images available on GitHub Container Registry
* **Modern Stack** - Built with FastAPI, ONNX Runtime, and Python 3.13+

Run `uv run python benchmark.py` to benchmark on your own hardware.

### Limitations

The current REST API is designed for [Photofield] integration. The CLIP model itself has some limitations as noted by OpenAI:

_CLIP currently struggles with respect to certain tasks such as fine grained classification and counting objects. CLIP also poses issues with regards to fairness and bias which we discuss in the paper and briefly in the next section._

See the [CLIP: Model Use] section for more details on responsible model usage.

### Built With

* [Python]
* [uv] - dependency management
* [FastAPI] - REST API framework
* [ONNX Runtime] - machine learning inference
* [CLIP Variants] - CLIP converted to ONNX (by yours truly)
* [UniFace] - face detection and analysis library
* [+ more Python libraries](pyproject.toml)

## Getting Started

Choose one of the following methods to run Photofield AI:

### Quick Start with Docker (Recommended)

The easiest way to get started is using Docker:

```bash
docker run -it -p 8081:8081 ghcr.io/smilyorg/photofield-ai:latest
```

The `clip-vit-base-patch32-(visual|textual)-float16` models are bundled for an out-of-the-box experience.

**Note:** The Docker image is currently CPU-only. GPU support contributions are welcome!

#### Verify it's running

Open [http://localhost:8081/docs](http://localhost:8081/docs) in your browser to see the API documentation.

#### Connect with Photofield

Add the following to your Photofield `configuration.yaml`:

```yaml
ai:
  # photofield-ai API server URL
  host: http://localhost:8081
```

### Install with uv (Development)

For development or customization, install from source using [uv]:

#### Prerequisites

- [Python] 3.13 or later
- [uv] (recommended) or pip

#### Quick Install

```bash
# Clone the repository
git clone https://github.com/smilyorg/photofield-ai.git
cd photofield-ai

# Install dependencies
uv sync

# Run the server
uv run python main.py
```

Or use [Task] for convenience:

```bash
task run    # Run the server
task watch  # Run with auto-reload
```

#### First Run

On first run, the server will download the default models (~300MB) and then start:

```
❯ uv run python main.py
Available providers: CPUExecutionProvider
Using providers: CPUExecutionProvider
Loading visual model: models/clip-vit-base-patch32-visual-float16.onnx
Loading textual model: models/clip-vit-base-patch32-textual-float16.onnx
Visual inference ready, input size 224, type tensor(float16)
Textual inference ready, input size 77, type tensor(int32)
Listening on 0.0.0.0:8081
```

### Build Your Own Docker Image

If you want to build the Docker image locally:

```bash
# Build the image
task docker

# Or manually
docker build -t photofield-ai .

# Run your local build
docker run -it -p 8081:8081 photofield-ai
```

# Usage

Some request/response examples are listed below. If you use the neat [REST Client] extension for VSCode you can even execute them directly if you open the README 😎. See [examples.http](examples.http) for more.

`{{api}}` refers to the root URL of the API, the following defines it for the [REST Client] extension.

```http
@api = http://localhost:8081
```

## Embed Text

The `/text-embeddings` endpoint accepts a list of text strings that are
converted to embeddings by the textual model.

### Request

```http
POST {{api}}/text-embeddings HTTP/1.1
Content-Type: application/json

{
    "texts": ["hawk"]
}
```

### Response

```json
{
  "texts": [
    {
      "text": "hawk",
      "embedding_f16_b64": "/7SvLO0xxqbgs4urVqq/vT21ozEAsGcu060XMxY0WyJbOBczIrJ0rkM4UB+Mrag007PhMuspxzTLshAxEjDgtXOsOzbrMRYwii2UKwS1IzNlsKUum6posPIwTS4KqG60arLbJgWu/K8hNaW1ry1QMx0hPpposuox2jMSsCixL6KNJQCw/LGttB4xwyoxM72QUa14NrsyMapespypirRHuHW177F5MNI0JiGfH+CxsbATMSqzIa94Mpy+eDcONnWvki27s4AqK7MlNsKgnjUpJ6Cy8y5snAqtTrB4JASxHCr4NF4wa6lMtHS2PK3HL1CtUjLKtXcuyTQlMQY0h7JZqU+1bTMpryYwArD7RSoonrR4rFEugTVaNssw8bBKKsyxzbKTLT22Jy8Rscy1BCmvsTSwT654pwkrii8gLkK116kDpvCwJ63frxYl9K+2sCk02DQ+LdS0C6dSrH0uOrUqL3e3vqa7sA4snjVgM48xeicYMMEoyy6AMAsjB69ft7OiDzC1MLKwtKbVIpqn6KjatmIsrjD8rY2xWa7MKW0fXLePuTWpdza/MYi5aB3atZivSqyYqC4kI60pKda0qbHbqLM126IFNA4xtrXKojyvDCynNGQioi4pLU01sp5/tcmoyasdODmuN7StLBawsqZKuYSsULSgtZE18i9SOLy5a7JksecrrS2dLGwx5q4etdm1rrQELLovxbY+prI0I7dLuXoxH66JKEkwszQitCSyia1cMfgwNK3lqDg0XzHlNrooTbhos9KyOR6GMYM1gbC8KyiwM7MoM+Oo7a7RLZ4mSx2wpSKy2qccMgEyFTS1tYMc+EWqNgayvzIfMrKrlDg3mV44GTBDKDg42jIOLBewtChhLNbAerFTNGywyDP3NjkxnTCgr8kyQrPoKLytxCo/tKYsQrDoJJSx2K7jNO8pjLL8KpirojVoMxWtV6msMjcyYDIqsIypYDSvNSO2ZK4+rg21EKE3owC0ALMSMwMu2KeKtXE0rbLPpsW20yUiLWuv/bUdsVuoKDYtKBSv9K4FJ560arXxNVs1RKwCsHCnkjhktam1rTGUn6yvdimYOKUHRDbHsHU38TdGKhE1UTKsLuYqxyVGq5UwaglCtEu0HiVasgg5ubQdrrUzLCw2Ji8yCa3ksnYpYzJlsF4y3beDKlaxFrRLnX+0W6IsOHO1TaE2s4gwKrEUNMWv5yFhsRosAzTVNtoxJrZKMCIoaavTM5UvMyyvLfM0D6zTtc60cK7Hlm62/iubLZgyODibrOgwN62auL0opzPiMvcvsK3KMpmtEbNIta+lYSysOM+vEDPqMOIzJKkOtCsq6bS3MqM3XixvsEC0Ni2HrbC1Bay4rg==",
      "embedding_inv_norm_f16_uint16": 11810
    }
  ]
}
```

* `embedding_f16_b64` - embedding that comes out of machine learning model. It's
  a base64-encoded list of 512 (or more, depending on model) float16 2-byte
  floating point values. You can compare this embedding to any other text or
  image embedding via [cosine similarity] (normalized dot product) to get the
  semantic similarity between them.

* `embedding_inv_norm_f16_uint16` - the [Euclidean / L2 norm][norm] of the
  embedding (vector length). It is inverted, converted to float16 2-byte
  floating point and then written out as an integer uint16 value. Using this
  precomputed value comes in handy while computing the [cosine similarity] for
  semantic image search as you can skip computing it for each image embedding.

## Embed Images

The `/image-embeddings` endpoint accepts multiple form multipart image uploads
and computes the embedding for each using the visual model.

```http
@image = heavy-industry.jpg
```

### Request

```http
POST {{api}}/image-embeddings
Content-Type: multipart/form-data; boundary=------------------------23f534be8db8eca0

--------------------------23f534be8db8eca0
Content-Disposition: form-data; name="image"; filename="heavy-industry.jpg"
Content-Type: image/jpeg

< {{image}}

--------------------------23f534be8db8eca0
```

### Response

```json
{
  "images": [
    {
      "field": "image",
      "filename": "heavy-industry.jpg",
      "embedding_f16_b64": "VykzsNCu7CQ2NLCqkjQjspWtlDeXrmm4z6htMZ6z7ayNO2M2xbVrqjU0gK3uN3c0hZiRMDY6jDTXm2GkUrZUrXarLzZINGonQ7ScuI64QTk+tH64LbhrNOU2BLU6tfYyjrT1t5QtYKgOMfi3b6lxty2437JWNA+x1TceMtQtNbBVqXegt6ttMMUtXLFaseo0H7BEt+U1bzF+tPGs0iAmsPAtCDFus/m0m64SNo8ujTDDJoIodi/vMY3Fa5meLMy3/LECOUWv/7QXIcOwO7o2K+03nLIev0A1YR1KNBkwnrbptuszlDWcrrkyOrP4Mdk7dzCGrk8tmTciMvWyti5iNGieNTMyMyWsJrmdOoAulyzGOHEvoLOYrE8nvy9TtQsym7HGJDS4MDVcsRsuyR2CN/wo7bIHODc4HDvPNWgiCy+kqkSkmrdtsuuyVDXhu+41r7FWLKc44a1rsFSx8bOJMnywRjLrNt4xNyYSpmgoLrSnsLA0KDUbq1gp7rSCLZkg+bExOIq01rAaqtuxM6yqtB4wELQ4sqC2UrYSrCe1WqzdtGOsEzbZqYE0oqBZMSo087VBNBO6UqorNuo0iLiktLO5gjYtNz+237LLomupHTgTMWkrQRyDPC80WjFQtCEwxjqaME23rytRr0E7Oy+SMma6+7B9ILwhaTjJM2Mwkjc+tKOwWT+SN5k2JKsAMu89LK4ttxqty7QVL7OuobUkolikITQ3trutbbHSGEK1cjQ5MwMszq5Itqq01LBzPFE0abfSJwe0XDB1t9M2orNytEg2QClbNW00Frmer9uz5qIVLPO0+CRxLUk1rTIOMBUwmzqrNlg3pjOGrz824bhutI40RT0stMS1yLK9LjKtwLAvuHevtzAlOioz/LEVuJa2CLgDMbY2PycplR00VLGPHvEws7gTMTMuxyh2NmI1srE6qFMse6yYtKm1QDUcs6Esz7jmsM2zUa8ltA4t/p0buGM2f6pHpw+127S7I4cwKa+1sMGhRbKiNWi06ytKtqCkh7Y7NoqxdCIJM+i3wLSYs+uvUTTPtFU1mSC4tE83UjMdMJitpzUZM7Qz8hn4tiMsbbJjK+m6BTDUsmc0+7MqLiI23DS2LO8mUCi+rB00zS1JrZKvKLDkM1S0mjCEOMqy3zeZKWk2mbD0uZWwDjXbpzU0/zoeMhS56qi5NIywXbilMi619TaCrrO3rawoqoOyOTHarBolarJFMGQySrf/sVWn5Tm0LXqvHLeCsssuVixJOF2uN5DMJi4vL7SSNGwqeLOEtHmtnThJOSowWjWsMEQ4bbiVK+qsmzNFuvwoM7amtjeqC7VWJh847rgvpIE0OToiuDSzE6R8M8U4wbQ1sA==",
      "embedding_inv_norm_f16_uint16": 11922
    }
  ]
}
```

## Detect Faces

The `/faces` endpoint accepts multipart image uploads and detects faces in each image, returning bounding boxes, confidence scores, and 5-point facial landmarks. This uses the [UniFace] library with RetinaFace detector.

### Request

```http
POST {{api}}/faces
Content-Type: multipart/form-data; boundary=------------------------23f534be8db8eca0

--------------------------23f534be8db8eca0
Content-Disposition: form-data; name="image0"; filename="faces.jpg"
Content-Type: image/jpeg

< faces.jpg

--------------------------23f534be8db8eca0
```

### Response

```json
{
  "images": [
    {
      "field": "image0",
      "filename": "photo.jpg",
      "faces": [
        {
          "bbox": [97.6, 447.04, 150.13, 510.99],
          "confidence": 0.9997,
          "landmarks": [
            [113.6, 472.39],
            [136.25, 470.5],
            [126.59, 486.55],
            [118.62, 498.68],
            [134.62, 497.35]
          ]
        }
      ]
    }
  ]
}
```

* `bbox` - Bounding box coordinates in [x1, y1, x2, y2] format
* `confidence` - Detection confidence score (0.0 to 1.0)
* `landmarks` - 5-point facial landmarks: left eye, right eye, nose, left mouth corner, right mouth corner

## Configuration

You can configure the app via environment variables.

| Environment variable name | Default value | Purpose |
| --- | --- | --- |
| `PHOTOFIELD_AI_HOST` | `0.0.0.0` | The host the server will listen on. |
| `PHOTOFIELD_AI_PORT` | `8081` | The port the server will listen on. |
| `PHOTOFIELD_AI_MODELS_DIR` | `models/` | The directory models will be downloaded to if a URL is provided |
| `PHOTOFIELD_AI_VISUAL_MODEL` | `https://huggingface.co/mlunar/clip-variants/resolve/main/modelclip-vit-base-patch32-visual-float16.onnx` | URL or local file path to the visual ONNX CLIP model to use for image embedding. If a URL is provided, the model will first be downloaded to `PHOTOFIELD_AI_MODELS_DIR` if it doesn't exist there already. If a local path is provided, the model will be used as is. |
| `PHOTOFIELD_AI_TEXTUAL_MODEL` | `https://huggingface.co/mlunar/clip-variants/resolve/main/modelclip-vit-base-patch32-textual-float16.onnx` | Same as `PHOTOFIELD_AI_VISUAL_MODEL`, but for the textual model used for text embedding. |
| `PHOTOFIELD_AI_RUNTIME` | `all` | `all` enables all available ONNX runtime providers, making use of any GPU or other accelerator device if you have the right [ONNX Runtime] prerequisites installed. `cpu` for CPU-only execution, which is faster to startup and develop with, but it is usually going to be ~10x slower than a GPU at inference. `cpu` is a shortcut for `PHOTOFIELD_AI_PROVIDERS=CPUExecutionProvider`. |
| `PHOTOFIELD_AI_PROVIDERS` | unset | If `PHOTOFIELD_AI_RUNTIME` is not set, you can use this specify the ONNX providers you would like to use directly comma-delimited. For example: `CUDAExecutionProvider,CPUExecutionProvider`. |

### Models

For `PHOTOFIELD_AI_VISUAL_MODEL` and `PHOTOFIELD_AI_TEXTUAL_MODEL` you can use
any model from [clip-variants models].

The bigger models are likely to be better, however it probably depends on your
use-case. The different model types most likely won't be compatible with each
other, however combining different data types might work fine.

Note that the `qint8` models don't seem to work right now, so use `quint8` ones
instead.

## Troubleshooting

### Server won't start

**Issue:** Server fails to start or crashes immediately

**Solutions:**
- Ensure you have Python 3.13 or later: `python --version`
- Make sure dependencies are installed: `uv sync`
- Check if port 8081 is already in use: `lsof -i :8081` (Linux/Mac) or `netstat -ano | findstr :8081` (Windows)

### Models not downloading

**Issue:** First run doesn't download models or fails partway through

**Solutions:**
- Check your internet connection
- Verify you have ~300MB of free disk space
- Try downloading models manually from [clip-variants models] and place them in the `models/` directory

### GPU not being used

**Issue:** Server runs but only uses CPU

**Solutions:**
- Verify GPU support: The server will print available providers on startup
- Install GPU-specific dependencies for your hardware (CUDA for NVIDIA, etc.)
- Set `PHOTOFIELD_AI_RUNTIME=all` to enable all available providers
- Check [ONNX Runtime] documentation for GPU prerequisites

### Out of Memory errors

**Issue:** Server crashes with memory errors during inference

**Solutions:**
- Use smaller models (e.g., `patch16` instead of `patch32`)
- Reduce batch sizes if processing multiple images
- Switch to CPU execution: `PHOTOFIELD_AI_RUNTIME=cpu`

### Connection issues with Photofield

**Issue:** Photofield can't connect to the AI server

**Solutions:**
- Verify the server is running: `curl http://localhost:8081/docs`
- Check the `host` URL in Photofield's `configuration.yaml`
- If using Docker, ensure the port mapping is correct: `-p 8081:8081`
- Check firewall settings if running on different machines

## Development

For development work on Photofield AI itself:

### Prerequisites

* [Python] 3.13 or later
* [uv] - for dependency management
* [Task] - for running development tasks

### Setup

```bash
git clone https://github.com/smilyorg/photofield-ai.git
cd photofield-ai
uv sync
```

### Available Tasks

Run `task` to see all available tasks, or use these common ones:

* `task run` - Run the server
* `task watch` - Run with auto-reload for development
* `task benchmark` - Run performance benchmarks
* `task docker` - Build Docker image
* `task sync` - Install/sync dependencies
* `task update` - Update dependencies to latest versions
* `task clean` - Clean build artifacts and cache

### Testing & Benchmarks

* Run `task benchmark` to benchmark performance on your hardware (requires server running in another terminal)
* Test API endpoints with example requests in [examples.http](examples.http) using the [REST Client] extension for VSCode

## Contributing

Pull requests are welcome. For major changes, please open an issue first to
discuss what you would like to change.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## Acknowledgements
* [OpenAI CLIP] for the research and machine learning model weights used here
* [Hugging Face](https://huggingface.co/) for hosting the ONNX models
* [CLIP-as-service by Jina](https://github.com/jina-ai/clip-as-service) as a big inspiration for this project
* [openai-clip-js by josephrocca](https://github.com/josephrocca/openai-clip-js) on how to convert CLIP to ONNX
* [CLIP-ONNX by Lednik7](https://github.com/Lednik7/CLIP-ONNX) on more CLIP with ONNX example code
* [Exporting a Model from PyTorch to ONNX and running it using ONNX Runtime - PyTorch](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
* [imgbeddings by minimaxir](https://github.com/minimaxir/imgbeddings) for a similar image-focused CLIP ONNX implementation
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)
* [readme.so](https://readme.so/)


[Photofield]: https://github.com/SmilyOrg/photofield
[OpenAI CLIP]: https://github.com/openai/CLIP/
[CLIP: Model Use]: https://github.com/openai/CLIP/blob/main/model-card.md#model-use
[Cosine similarity]: https://en.wikipedia.org/wiki/Cosine_similarity
[norm]: https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm

[Python]: https://www.python.org/
[Git]: https://git-scm.com/downloads
[uv]: https://docs.astral.sh/uv/
[FastAPI]: https://fastapi.tiangolo.com/
[ONNX Runtime]: https://onnxruntime.ai/
[CLIP Variants]: https://huggingface.co/mlunar/clip-variants
[UniFace]: https://github.com/yakhyo/uniface
[clip-variants models]: https://huggingface.co/mlunar/clip-variants/tree/main/models
[REST Client]: https://marketplace.visualstudio.com/items?itemName=humao.rest-client
[Task]: https://taskfile.dev/

[Configuration]: #configuration

[open an issue]: https://github.com/SmilyOrg/photofield-ai/issues
[Getting Started]: #getting-started
