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
    <li><a href="#development-setup">Development Setup</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



## About

Photofield AI is a companion to [Photofield] providing AI features. It's a
separate REST API service both to keep the main app slim and because AI features
are currently easier to implement in Python as opposed to Go. It is an API
currently exposing the [OpenAI CLIP] image and text embedding functionality.

### Features

Returns [OpenAI CLIP] images and text embeddings that you can then compare with
[Cosine similarity] for use in semantic image search. Image embedding runs at up
to ~20 requests/sec on an i7-5820K CPU and up to ~200 requests/sec using a
GeForce GTX 1070 Ti. Resource utilization with a GPU is low, so I imagine there
are some bottlenecks in some parts of the system, but 200 requests/sec seems
plenty enough as is.

### Limitations

The current REST API is tied pretty closely to [Photofield]. The machine
learning model itself also has some limitations and bias, as was reported by
OpenAI:

_CLIP and our analysis of it have a number of limitations. CLIP currently
struggles with respect to certain tasks such as fine grained classification and
counting objects. CLIP also poses issues with regards to fairness and bias which
we discuss in the paper and briefly in the next section._

See more on model use in the [CLIP: Model Use] section of the model card from OpenAI.

### Built With

* [Python]
* [FastAPI] - REST API framework
* [ONNX Runtime] - machine learning inference
* [CLIP Variants] - CLIP converted to ONNX (by yours truly)
* [+ more Python libraries](pyproject.toml)

## Getting Started

### Docker

Coming soon™.

### From Source

#### Prerequisites

1. [Python]
2. [Poetry]

#### Setup

1. [Download the
   source](https://github.com/SmilyOrg/photofield-ai/archive/refs/heads/main.zip)
   or clone the Git repository
2. In the source directory you downloaded, run `poetry install` to install the
   required dependencies. You can also run `poetry install --without gpu` to
   skip installing GPU dependencies if you want to run it on CPU only (it is
   also a smaller install).
3. After [Poetry] installs all the required dependencies, the server should be
   ready to run.

#### Run

Run the server with `poetry run python main.py`. If you don't specify any model
files, it should first download the default models and then start listening to
requests.

```
❯ poetry run python main.py
Available providers: TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider
Using providers: TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider
Loading visual model: models/clip-vit-base-patch32-visual-float16.onnx
Loading textual model: models/clip-vit-base-patch32-textual-float16.onnx
2022-10-08 14:28:35.9706571 [W:onnxruntime:Default, tensorrt_execution_provider.h:60 onnxruntime::TensorrtLogger::log] [2022-10-08 13:28:35 WARNING] external\onnx-tensorrt\onnx2trt_utils.cpp:369: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.

Visual inference ready, input size 224, type tensor(float16)
Textual inference ready, input size 77, type tensor(int32)
Listening on 0.0.0.0:8081
```

If you are starting it with GPU support (default) it may take some time for it to start up. The `WARNING` above is to be expected for the TensorRT runtime, it seems to work fine regardless.

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

## Development Setup

### Prerequisites

* [Python]
* [Poetry] - for dependency management
* [just] - to run common commands conveniently
* sh-like shell (e.g. sh, bash, busybox) - required by `just`

**[Scoop] (Windows)**: `scoop install busybox just`

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/smilyorg/photofield-ai.git
   ```
2. Install Python dependencies
   ```sh
   poetry install
   ```

### Running

* `poetry shell` to enter the virtual environment and `just watch` the source
  files and auto-reload the server
* or `just run` the server

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
[Poetry]: https://python-poetry.org/docs/#installation
[FastAPI]: https://fastapi.tiangolo.com/
[ONNX Runtime]: https://onnxruntime.ai/
[CLIP Variants]: https://huggingface.co/mlunar/clip-variants
[clip-variants models]: https://huggingface.co/mlunar/clip-variants/tree/main/models
[REST Client]: https://marketplace.visualstudio.com/items?itemName=humao.rest-client

[Configuration]: #configuration

[open an issue]: https://github.com/SmilyOrg/photofield-ai/issues
[Getting Started]: #getting-started

[Scoop]: https://scoop.sh/
[just]: https://github.com/casey/just
[watchexec]: https://github.com/watchexec/watchexec
