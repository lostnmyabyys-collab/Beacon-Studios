# Beacon Studios AI Platform

A production-grade, modular AI model ecosystem supporting the VX, NEO, and CZ model families. Built for enterprise-scale distributed training, inference, and deployment while remaining development-friendly.

## Overview

Beacon Studios provides a complete foundation for:

- **Training**: Enterprise-grade distributed training pipeline with automatic checkpointing, mixed precision, and validation
- **Inference**: Custom high-performance runtime supporting batch processing, streaming, and multi-model orchestration
- **Datasets**: Comprehensive dataset management with automatic cleaning, deduplication, and versioning
- **Evaluation**: Automated benchmarking across multiple dimensions with comparison reports
- **APIs**: REST and WebSocket APIs for model access and administration
- **Desktop Integration**: Hooks for Beacon AI desktop application integration
- **Model Registry**: Centralized model metadata, versioning, and discovery

## Model Families

### VX Series
Fast, lightweight assistant models optimized for latency. Ideal for chat, writing, summarization, and general reasoning.
- VX-mini, VX-1, VX-2

### NEO Series
Advanced reasoning models with large context windows. Optimized for research, mathematics, complex analysis, and long-form generation.
- NEO-mini, NEO-1, NEO-2, NEO-Ultra

### CZ Series
Professional software engineering models with repository understanding, code generation, debugging, and agent workflows.
- CZ-mini, CZ-1, CZ-2, CZ-Pro

## Project Structure

```
beacon-studios/
├── beacon/                          # Main package
│   ├── runtime/                     # Inference runtime
│   │   ├── engine.py
│   │   ├── scheduler.py
│   │   ├── cache.py
│   │   ├── batch.py
│   │   └── streaming.py
│   ├── training/                    # Training pipeline
│   │   ├── trainer.py
│   │   ├── optimizer.py
│   │   ├── scheduler.py
│   │   ├── validation.py
│   │   └── distributed.py
│   ├── datasets/                    # Dataset management
│   │   ├── manager.py
│   │   ├── loader.py
│   │   ├── cleaner.py
│   │   └── builders/
│   │       ├── conversation.py
│   │       ├── instruction.py
│   │       ├── code.py
│   │       └── reasoning.py
│   ├── tokenizer/                   # Tokenizer management
│   │   ├── manager.py
│   │   ├── trainer.py
│   │   └── vocab.py
│   ├── models/                      # Model architecture
│   │   ├── config.py
│   │   ├── transformer.py
│   │   ├── attention.py
│   │   ├── embeddings.py
│   │   └── families/
│   │       ├── vx.py
│   │       ├── neo.py
│   │       └── cz.py
│   ├── registry/                    # Model registry
│   │   ├── manager.py
│   │   ├── metadata.py
│   │   └── versioning.py
│   ├── checkpoints/                 # Checkpoint management
│   │   ├── manager.py
│   │   ├── saver.py
│   │   └── loader.py
│   ├── evaluation/                  # Evaluation & benchmarks
│   │   ├── suite.py
│   │   ├── benchmarks/
│   │   │   ├── reasoning.py
│   │   │   ├── coding.py
│   │   │   ├── math.py
│   │   │   └── instruction.py
│   │   └── reporter.py
│   ├── api/                         # API layer
│   │   ├── rest/
│   │   │   ├── app.py
│   │   │   ├── routes/
│   │   │   │   ├── models.py
│   │   │   │   ├── inference.py
│   │   │   │   ├── training.py
│   │   │   │   ├── datasets.py
│   │   │   │   └── admin.py
│   │   │   └── middleware.py
│   │   └── websocket/
│   │       ├── app.py
│   │       ├── handlers.py
│   │       └── streaming.py
│   ├── auth/                        # Authentication
│   │   ├── manager.py
│   │   ├── tokens.py
│   │   └── policies.py
│   ├── config/                      # Configuration
│   │   ├── settings.py
│   │   ├── loader.py
│   │   └── schema.py
│   ├── storage/                     # Storage abstraction
│   │   ├── base.py
│   │   ├── local.py
│   │   ├── s3.py
│   │   └── manager.py
│   ├── logging/                     # Logging & metrics
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   └── telemetry.py
│   ├── desktop/                     # Desktop integration
│   │   ├── integration.py
│   │   ├── hooks.py
│   │   └── ipc.py
│   └── utils/                       # Utilities
│       ├── decorators.py
│       ├── validators.py
│       ├── serialization.py
│       └── device.py
├── tests/                           # Test suite
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
├── models/                          # Model definitions
│   ├── vx/
│   ├── neo/
│   └── cz/
├── scripts/                         # Utility scripts
│   ├── setup.py
│   ├── download_models.py
│   ├── train.py
│   └── evaluate.py
├── docker/                          # Docker configuration
│   ├── Dockerfile
│   ├── Dockerfile.dev
│   └── docker-compose.yml
├── docs/                            # Documentation
│   ├── ARCHITECTURE.md
│   ├── TRAINING.md
│   ├── INFERENCE.md
│   ├── API.md
│   └── DEPLOYMENT.md
├── .github/workflows/               # CI/CD
│   ├── test.yml
│   ├── build.yml
│   └── deploy.yml
├── pyproject.toml
├── requirements.txt
└── .env.example
```

## Quick Start

### Development Setup

```bash
# Clone repository
git clone https://github.com/lostnmyabyys-collab/Beacon-Studios.git
cd Beacon-Studios

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e .

# Set up environment
cp .env.example .env
```

### Training a Model

```python
from beacon.training import Trainer
from beacon.datasets import DatasetManager
from beacon.config import load_config

# Load configuration
config = load_config("config/vx_mini.yaml")

# Initialize trainer
trainer = Trainer(config)

# Train
trainer.train()
```

### Running Inference

```python
from beacon.runtime import BeaconRuntime

# Initialize runtime
runtime = BeaconRuntime()

# Load model
model = runtime.load_model("vx-1", version="1.0.0")

# Generate
response = model.generate(
    prompt="Hello, how are you?",
    max_tokens=256,
    temperature=0.7
)
```

### REST API

```bash
# Start server
python -m beacon.api.rest.app

# Example request
curl -X POST http://localhost:8000/api/v1/inference/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vx-1",
    "prompt": "Explain quantum computing",
    "max_tokens": 512
  }'
```

## Architecture Principles

1. **Modularity**: Clear separation of concerns with well-defined interfaces
2. **Scalability**: Design scales from single GPU to distributed systems
3. **Production-Ready**: No placeholders; all implementations are complete
4. **Type Safety**: Full type hints throughout the codebase
5. **Testability**: Comprehensive unit and integration tests
6. **Documentation**: Inline docs, architecture guides, and API references
7. **Extensibility**: Plugin architecture for custom components
8. **Security**: Secure defaults, authentication, and access control

## Technology Stack

- **Python 3.12+** for core implementation
- **PyTorch** for model training and inference
- **Transformers** for model architectures and utilities
- **Tokenizers** for efficient tokenization
- **FastAPI** for REST API
- **WebSockets** for real-time communication
- **PostgreSQL** for metadata storage
- **Redis** for caching and queues
- **Docker** for containerization
- **GitHub Actions** for CI/CD

## Performance Targets

- **VX Models**: <100ms latency at P99
- **NEO Models**: <500ms latency at P99
- **CZ Models**: <300ms latency at P99
- **Training**: Distributed training across multiple GPUs
- **Throughput**: Batch processing with dynamic batching
- **Memory**: Efficient KV caching and quantization support

## Supported Features

- Mixed precision training (FP32, FP16, BF16)
- Distributed training (DDP, FSDP)
- Quantization (INT8, FP8, ONNX export)
- Streaming generation
- Batch inference
- Multi-model orchestration
- Checkpoint management and resumption
- Automatic validation and early stopping
- Comprehensive benchmarking

## Configuration

All systems are configured via YAML files in the `config/` directory:

- `config/training.yaml` - Training hyperparameters
- `config/inference.yaml` - Runtime settings
- `config/datasets.yaml` - Dataset sources and processing
- `config/models/` - Model family configurations

See [CONFIG.md](docs/CONFIG.md) for detailed configuration options.

## API Documentation

### REST API
- `/api/v1/models` - Model discovery and management
- `/api/v1/inference/generate` - Text generation
- `/api/v1/training/submit` - Training job submission
- `/api/v1/datasets` - Dataset management
- `/api/v1/benchmarks` - Evaluation results

### WebSocket API
- `ws://localhost:8000/ws/inference/stream` - Streaming inference
- `ws://localhost:8000/ws/training/monitor` - Training monitoring

See [API.md](docs/API.md) for full documentation.

## Development

### Running Tests

```bash
pytest tests/unit -v
pytest tests/integration -v
pytest tests/benchmarks -v
```

### Code Quality

```bash
black beacon/
isort beacon/
pylint beacon/
mypy beacon/
```

### Building Docker Images

```bash
docker-compose -f docker/docker-compose.yml build
docker-compose -f docker/docker-compose.yml up
```

## Roadmap

- [x] Project structure and foundation
- [x] Core runtime and inference engine
- [x] Training pipeline
- [x] Dataset management
- [x] Model registry
- [ ] Advanced evaluation suite
- [ ] Desktop integration
- [ ] Multi-node distributed training
- [ ] Advanced caching strategies
- [ ] Model quantization pipeline

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

Proprietary - Beacon Studios

## Support

For issues and questions, please open an issue on GitHub.

---

**Beacon Studios AI Platform** | Built for scale, designed for simplicity
