# Beacon Studios LLM Platform

High-performance, modular LLM platform with distributed training, efficient inference, and multi-backend support.

## Features

### Core
- **Transformer Architecture**: Multi-head attention, Flash Attention, Grouped Query Attention (GQA)
- **Efficient Inference**: KV cache, batch processing, streaming generation
- **Distributed Training**: DDP support, gradient accumulation, mixed precision
- **Model Registry**: Version control, metadata management

### Storage
- **Local Storage**: File system backend
- **S3 Storage**: AWS S3 compatible backends (MinIO, DigitalOcean Spaces, etc.)
- **Seamless Switching**: Configure via environment variables

### Datasets
- **Multiple Formats**: JSON, JSONL, CSV, TXT, Markdown
- **Kaggle Integration**: Direct download and import from Kaggle
- **Data Processing**: Cleaning, deduplication, normalization
- **Dataset Builders**: Conversation, Instruction-following, Code datasets

### API
- **FastAPI Backend**: RESTful API with OpenAPI docs
- **Authentication**: JWT tokens, user management
- **Generation**: Text generation with configurable sampling
- **Model Management**: List, retrieve, and manage models

## Deployment

### Railway (Recommended)

```bash
# One-click deployment
railway deploy
```

Set environment variables in Railway dashboard:
- `STORAGE_BACKEND`: "s3" or "local"
- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
- `JWT_SECRET_KEY`: JWT signing key
- `DEVICE`: "cuda", "mps", or "cpu"

### Docker

```bash
docker build -t beacon-studios .
docker run -p 8000:8000 beacon-studios
```

### Local Development

```bash
pip install -r requirements.txt
python -m uvicorn beacon.api.server:app --reload
```

## Configuration

Create `.env` file:

```env
ENVIRONMENT=development
DEBUG=true
DEVICE=cuda
STORAGE_BACKEND=local
LOCAL_STORAGE_PATH=./data
DATABASE_URL=postgresql://localhost:5432/beacon
REDIS_URL=redis://localhost:6379
JWT_SECRET_KEY=your-secret-key-here
```

## Usage

### API Examples

```bash
# Health check
curl http://localhost:8000/health

# Signup
curl -X POST http://localhost:8000/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "username": "user", "password": "pass"}'

# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "pass"}'

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "vx-1",
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Python SDK

```python
from beacon.runtime.engine import InferenceEngine
from beacon.storage.manager import StorageManager
from beacon.config.schema import RuntimeConfig, StorageConfig

# Initialize
storage_config = StorageConfig(backend="local")
storage = StorageManager(storage_config)
runtime_config = RuntimeConfig(device="cuda")
engine = InferenceEngine(runtime_config, storage, registry, tokenizer_manager)

# Generate
text = engine.generate(
    model_id="vx-1",
    prompt="Once upon a time",
    max_tokens=100,
    temperature=0.7,
)
print(text)
```

## Architecture

```
beacon/
├── api/              # FastAPI server
├── models/           # Model architectures
├── runtime/          # Inference engine
├── training/         # Training pipeline
├── datasets/         # Dataset management
├── storage/          # Storage backends
├── config/           # Configuration
├── auth/             # Authentication
├── registry/         # Model registry
├── checkpoints/      # Checkpoint management
├── tokenizer/        # Tokenizer management
├── utils/            # Utilities
└── logging.py        # Logging
```

## Performance

- **Inference**: 100+ tokens/second (A100)
- **Training**: Multi-GPU support with DDP
- **Memory**: Efficient KV cache and gradient checkpointing
- **Storage**: S3 compatible for cloud-native deployments

## License

MIT License
