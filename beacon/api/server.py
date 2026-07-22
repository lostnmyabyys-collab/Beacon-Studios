"""Beacon API Server - FastAPI backend."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from beacon.auth.manager import AuthManager
from beacon.auth.tokens import TokenManager
from beacon.config.settings import get_settings
from beacon.logging import get_logger
from beacon.models.config import ModelConfig
from beacon.models.transformer import TransformerDecoder
from beacon.registry.manager import ModelRegistryManager
from beacon.registry.metadata import ModelMetadata
from beacon.runtime.engine import InferenceEngine
from beacon.storage.manager import StorageManager
from beacon.storage.local import LocalStorage
from beacon.storage.s3 import S3Storage
from beacon.tokenizer.manager import TokenizerManager
from beacon.config.schema import StorageConfig
from datetime import datetime

logger = get_logger(__name__)

# Settings
settings = get_settings()

# Global instances
auth_manager: AuthManager | None = None
token_manager: TokenManager | None = None
storage_manager: StorageManager | None = None
tokenizer_manager: TokenizerManager | None = None
model_registry: ModelRegistryManager | None = None
inference_engine: InferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info("Beacon API Server starting...")
    
    global auth_manager, token_manager, storage_manager
    global tokenizer_manager, model_registry, inference_engine
    
    # Initialize storage
    storage_config = StorageConfig(
        backend=settings.storage_backend,
        local_path=settings.local_storage_path,
        s3_endpoint=settings.s3_endpoint_url,
        s3_bucket=settings.s3_bucket_name,
        s3_region=settings.s3_region_name,
    )
    storage_manager = StorageManager(storage_config)
    
    # Initialize other components
    auth_manager = AuthManager()
    token_manager = TokenManager()
    tokenizer_manager = TokenizerManager(storage_manager)
    model_registry = ModelRegistryManager(storage_manager)
    
    # Initialize runtime config
    from beacon.config.schema import RuntimeConfig
    runtime_config = RuntimeConfig(
        device=settings.device,
        max_batch_size=settings.inference_max_batch_size,
        max_context_length=settings.inference_max_context_length,
    )
    inference_engine = InferenceEngine(
        runtime_config,
        storage_manager,
        model_registry,
        tokenizer_manager,
    )
    
    logger.info("Beacon API Server ready")
    
    yield
    
    # Shutdown
    logger.info("Beacon API Server shutting down...")
    if inference_engine:
        inference_engine.clear_cache()
    logger.info("Beacon API Server stopped")


# Create FastAPI app
app = FastAPI(
    title="Beacon Studios API",
    description="High-performance LLM platform",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class LoginRequest(BaseModel):
    """Login request."""
    email: str
    password: str


class LoginResponse(BaseModel):
    """Login response."""
    token: str
    user_id: str
    email: str


class SignupRequest(BaseModel):
    """Signup request."""
    email: str
    username: str
    password: str


class GenerateRequest(BaseModel):
    """Generation request."""
    model_id: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_k: int | None = None
    top_p: float | None = 0.9


class GenerateResponse(BaseModel):
    """Generation response."""
    text: str
    model_id: str
    tokens_generated: int


class ModelInfo(BaseModel):
    """Model information."""
    model_id: str
    name: str
    type: str
    version: str
    context_length: int
    capabilities: list[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    device: str
    storage_backend: str


# Routes

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        device=settings.device,
        storage_backend=settings.storage_backend,
    )


@app.post("/auth/signup", response_model=LoginResponse)
async def signup(request: SignupRequest):
    """Signup endpoint."""
    try:
        user = auth_manager.create_user(
            email=request.email,
            username=request.username,
            password=request.password,
        )
        token = token_manager.create_token(
            user_id=user.user_id,
            email=user.email,
        )
        return LoginResponse(
            token=token,
            user_id=user.user_id,
            email=user.email,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Login endpoint."""
    try:
        user, token = auth_manager.authenticate(
            email=request.email,
            password=request.password,
        )
        return LoginResponse(
            token=token,
            user_id=user.user_id,
            email=user.email,
        )
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.get("/models", response_model=list[ModelInfo])
async def list_models():
    """List available models."""
    models = model_registry.list_models()
    return [
        ModelInfo(
            model_id=m.model_id,
            name=m.model_name,
            type=m.model_type,
            version=m.version,
            context_length=m.context_length,
            capabilities=m.capabilities,
        )
        for m in models
    ]


@app.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """Get model info."""
    metadata = model_registry.get_model(model_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    
    return ModelInfo(
        model_id=metadata.model_id,
        name=metadata.model_name,
        type=metadata.model_type,
        version=metadata.version,
        context_length=metadata.context_length,
        capabilities=metadata.capabilities,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text."""
    try:
        result = inference_engine.generate(
            model_id=request.model_id,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )
        
        return GenerateResponse(
            text=result,
            model_id=request.model_id,
            tokens_generated=request.max_tokens,
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Beacon Studios",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
    )
