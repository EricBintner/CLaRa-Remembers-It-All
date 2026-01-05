# CLaRa-Remembers-It-All

🧠 **Production-ready inference server for Apple's CLaRA context compression model.**

> *"Because CLaRa remembers it all... in 16x less space."*

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

---

## What Is This?

**CLaRa-Remembers-It-All** is a standalone HTTP server that provides **semantic context compression** for RAG (Retrieval-Augmented Generation) systems.

You send it a list of memories/documents and a query → it compresses them into a dense representation and returns an answer, using **16x to 128x fewer tokens** than the original text while preserving meaning.

```
┌─────────────────┐         ┌─────────────────────┐         ┌──────────────┐
│  Your RAG App   │  HTTP   │  CLaRa-Remembers-   │         │   Answer +   │
│  (any language) │ ──────► │     It-All Server   │ ──────► │  Compression │
│                 │  POST   │                     │         │    Stats     │
└─────────────────┘         └─────────────────────┘         └──────────────┘
     memories[]                   CLaRA Model                  "User enjoys
     + query                      (7B params)                   hiking..."
```

**Key point:** This is a *universal tool* - it works with any RAG system, any programming language, any framework. Just make HTTP calls.

---

## Why Does This Exist?

### The Problem

RAG systems retrieve documents to provide context to LLMs, but:

1. **Context windows fill up fast** - 10 retrieved documents × 500 tokens = 5,000 tokens consumed
2. **More context ≠ better answers** - LLMs struggle with long, noisy contexts
3. **Cost scales with tokens** - API costs grow linearly with context size
4. **Latency increases** - More tokens = slower inference

### The Solution: CLaRA

Apple's [CLaRA](https://github.com/apple/ml-clara) (Continuous Latent Reasoning) compresses documents into **dense semantic representations** that preserve meaning while dramatically reducing token count:

| Compression Level | Token Reduction | Use Case |
|-------------------|-----------------|----------|
| `compression-16` | **16x smaller** | Best quality, recommended |
| `compression-128` | **128x smaller** | Maximum compression |

**Example:** 20 memories totaling 2,000 tokens → compressed to ~125 tokens (16x) → LLM answers from compressed context.

### Why This Server?

Apple released the CLaRA model weights but **no production server**. This project provides:

- **REST API** - Any app can use it via HTTP
- **Network offloading** - Run on a powerful machine, access from anywhere
- **Multi-backend** - CUDA, Apple Silicon, CPU support
- **Production-ready** - Health checks, Docker, configuration

---

## How It Works

### 1. Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    CLaRa-Remembers-It-All                      │
├────────────────────────────────────────────────────────────────┤
│  FastAPI Server (REST API)                                     │
│    ├── POST /compress  → Compress memories, generate answer    │
│    ├── GET  /status    → Model info, stats                     │
│    └── GET  /health    → Health check for load balancers       │
├────────────────────────────────────────────────────────────────┤
│  Model Layer                                                   │
│    ├── PyTorch Backend (CUDA/MPS)                              │
│    ├── MLX Backend (Apple Silicon native) [planned]            │
│    └── CPU Fallback                                            │
├────────────────────────────────────────────────────────────────┤
│  CLaRA Model (apple/CLaRa-7B-Instruct)                         │
│    └── 7B parameter model with compression layers              │
└────────────────────────────────────────────────────────────────┘
```

### 2. The Compression Process

When you call `/compress`:

1. **Input:** List of memory strings + query
2. **Encode:** Each memory is encoded into dense vectors
3. **Compress:** CLaRA's compression layers reduce 16-128 tokens → 1 token
4. **Generate:** Model generates answer from compressed representation
5. **Output:** Answer + compression statistics

### 3. Use Cases

- **Personal AI assistants** - Compress user history/preferences
- **Document Q&A** - Compress retrieved passages before answering
- **Chatbots with memory** - Store more context in less space
- **Cost optimization** - Reduce API token costs by 16x
- **Edge deployment** - Fit more context on smaller models

---

## Features

- 🔥 **FastAPI REST API** - Simple `/compress` endpoint
- 🐳 **Docker ready** - One command deployment
- 🍎 **Apple Silicon** - MPS backend, MLX coming soon
- 🖥️ **NVIDIA CUDA** - Full GPU acceleration
- 📊 **Production features** - Health checks, metrics, configurable
- 🌐 **Universal** - Use with any RAG system, any language
- 🔒 **Optional auth** - API key authentication
- ⚙️ **Configurable** - Environment variables for all settings

## Quick Start

### Docker (Recommended)

```bash
# NVIDIA GPU
docker run -p 8765:8765 --gpus all ghcr.io/ericbintner/clara-remembers-it-all

# Apple Silicon (coming soon)
docker run -p 8765:8765 ghcr.io/ericbintner/clara-remembers-it-all:mlx
```

### From Source

```bash
git clone https://github.com/ericbintner/CLaRa-Remembers-It-All.git
cd CLaRa-Remembers-It-All
pip install -e .
clara-server
```

## API Usage

### Compress Memories

```bash
curl -X POST http://localhost:8765/compress \
  -H "Content-Type: application/json" \
  -d '{
    "memories": [
      "User likes hiking in national parks.",
      "User works as a software engineer.",
      "User has a dog named Max."
    ],
    "query": "What outdoor activities does the user enjoy?"
  }'
```

**Response:**
```json
{
  "success": true,
  "answer": "The user enjoys hiking in national parks.",
  "original_tokens": 25,
  "compressed_tokens": 1,
  "compression_ratio": 16.0,
  "latency_ms": 342
}
```

### Check Status

```bash
curl http://localhost:8765/status
```

### Health Check

```bash
curl http://localhost:8765/health
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `CLARA_MODEL` | `apple/CLaRa-7B-Instruct` | HuggingFace model ID |
| `CLARA_SUBFOLDER` | `compression-16` | Compression level (16 or 128) |
| `CLARA_PORT` | `8765` | Server port |
| `CLARA_HOST` | `0.0.0.0` | Bind address |
| `CLARA_BACKEND` | `auto` | Backend: `auto`, `cuda`, `mps`, `mlx`, `cpu` |
| `CLARA_CACHE` | `~/.cache/clara-server` | Model cache directory |

## Backends

| Backend | Platform | VRAM Required | Status |
|---------|----------|---------------|--------|
| **CUDA** | Linux/Windows + NVIDIA | ~14GB | ✅ Stable |
| **MPS** | macOS + Apple Silicon | ~14GB unified | ✅ Stable |
| **MLX** | macOS + Apple Silicon | ~14GB unified | 🔜 Coming |
| **CPU** | Any | ~28GB RAM | ⚠️ Slow |

## Integration Examples

### Python Client

```python
import requests

def compress_memories(memories: list, query: str, url: str = "http://localhost:8765"):
    response = requests.post(f"{url}/compress", json={
        "memories": memories,
        "query": query
    })
    return response.json()

# Example
result = compress_memories(
    memories=["User likes coffee.", "User works remotely."],
    query="What does the user prefer?"
)
print(result["answer"])
```

### JavaScript/TypeScript

```typescript
async function compressMemories(memories: string[], query: string): Promise<any> {
  const response = await fetch("http://localhost:8765/compress", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ memories, query })
  });
  return response.json();
}
```

### LangChain Integration

```python
from langchain.retrievers import BaseRetriever

class ClaraCompressedRetriever(BaseRetriever):
    clara_url: str = "http://localhost:8765"
    
    def _get_relevant_documents(self, query: str):
        # Your retrieval logic + clara compression
        pass
```

## Deployment

### Docker Compose

```yaml
version: "3.8"
services:
  clara:
    image: ghcr.io/ericbintner/clara-remembers-it-all
    ports:
      - "8765:8765"
    volumes:
      - clara-cache:/root/.cache/clara-server
    environment:
      - CLARA_MODEL=apple/CLaRa-7B-Instruct
      - CLARA_SUBFOLDER=compression-16
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  clara-cache:
```

### Kubernetes

See [docs/kubernetes.md](docs/kubernetes.md) for Helm chart and deployment manifests.

## Requirements

- **CUDA**: NVIDIA GPU with 14GB+ VRAM (RTX 3090, RTX 4090, A100, etc.)
- **Apple Silicon**: Mac with 16GB+ unified memory (M1 Pro/Max, M2, M3, Mac Studio)
- **CPU**: 28GB+ RAM (not recommended for production)
- Python 3.10+

## Development

```bash
# Clone
git clone https://github.com/ericbintner/CLaRa-Remembers-It-All.git
cd CLaRa-Remembers-It-All

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest

# Run server in dev mode
clara-server --reload
```

## Roadmap

- [x] Core FastAPI server
- [x] CUDA backend
- [x] MPS backend (Apple Silicon via PyTorch)
- [x] Docker deployment
- [ ] MLX backend (native Apple Silicon)
- [ ] Batching for higher throughput
- [ ] Prometheus metrics
- [ ] GPTQ/AWQ quantization support
- [ ] Multi-GPU support
- [ ] Kubernetes Helm chart

## Known Issues

### bitsandbytes Quantization

4-bit and 8-bit quantization via bitsandbytes currently causes device mismatch errors with CLaRA's custom model architecture. We're tracking this issue and will add support when resolved. See [QUANTIZATION.md](docs/QUANTIZATION.md) for details.

**Current workaround:** Use fp16 with sufficient VRAM, or offload to a machine with more memory.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## License

Apache 2.0 - Same as [Apple's ml-clara](https://github.com/apple/ml-clara).

## Acknowledgments

- [Apple ML Research](https://github.com/apple/ml-clara) for the CLaRA model
- [HuggingFace](https://huggingface.co/apple) for model hosting
- The open-source RAG community

## Citation

If you use clara-server in your research, please cite the original CLaRA paper:

```bibtex
@article{clara2024,
  title={CLaRa: Bridging Retrieval and Generation with Continuous Latent Reasoning},
  author={Apple ML Research},
  year={2024}
}
```
