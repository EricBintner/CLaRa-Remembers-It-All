# CLaRa-Remembers-It-All

🧠 **Production-ready inference server for Apple's CLaRA context compression model.**

> *"Because CLaRa remembers it all... in 16x less space."*

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

## Why CLaRa-Remembers-It-All?

Apple's [CLaRA](https://github.com/apple/ml-clara) achieves **16x-128x semantic compression** for RAG systems, dramatically reducing context length while preserving meaning. However, there's no production-ready inference server.

**CLaRa-Remembers-It-All fills that gap:**

- 🔥 **FastAPI-based REST API** - Simple `/compress` endpoint
- 🐳 **Docker ready** - One command deployment
- 🍎 **Apple Silicon native** - MLX backend for Mac Studio/Pro
- 🖥️ **NVIDIA CUDA** - Full GPU acceleration
- 📊 **Production features** - Health checks, metrics, batching
- 🌐 **Universal** - Use with any RAG system, not framework-specific

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
