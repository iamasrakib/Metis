"""
Μῆτις (Metis) — REST API Server
==================================
FastAPI-based inference server with:
  • /generate — single text generation
  • /chat — interactive chat endpoint
  • /stream — Server-Sent Events streaming generation
  • /info — model metadata
  • OpenAI-compatible /v1/completions endpoint
  • OpenAI-compatible /v1/chat/completions endpoint
"""

import os
import sys
import json
import time
import logging
from typing import Optional, List, Dict, Any, Union
import threading
from contextlib import asynccontextmanager
from queue import Queue, Empty

import torch
import uvicorn
from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException, Query, Request
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
    print("❌ fastapi not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)

from metis import (
    MetisLM, ModelConfig, BPETokenizer, CharTokenizer,
    generate_text, load_model_and_tokenizer,
)

logger = logging.getLogger("metis.server")

# ── Global model state ───────────────────────────────────────────────────────
model_globals = {"model": None, "tokenizer": None, "config": None}


# ── Request / Response Models ─────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt to generate from")
    max_tokens: int = Field(200, ge=1, le=4096, description="Max tokens to generate")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(40, ge=0, description="Top-k filtering")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling")
    repetition_penalty: float = Field(1.1, ge=1.0, le=5.0, description="Repetition penalty")
    stream: bool = Field(False, description="Stream tokens via SSE")
    stop: Optional[str] = Field(None, description="Stop string")


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: user/assistant/system")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    max_tokens: int = Field(200, ge=1, le=4096)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(40)
    top_p: Optional[float] = Field(0.9)
    stream: bool = Field(False)


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    generation_time_ms: float


class ModelInfo(BaseModel):
    name: str = "Μῆτις (Metis)"
    version: str = "3.0"
    parameters: str = ""
    tokenizer: str = ""
    device: str = ""
    max_seq_len: int = 0
    vocab_size: int = 0
    dtype: str = ""


# ── OpenAI-compatible schemas ─────────────────────────────────────────────────

class OpenAICompletionRequest(BaseModel):
    model: str = "metis-3.0"
    prompt: Union[str, List[str]] = ""
    max_tokens: int = 200
    temperature: float = 0.8
    top_p: float = 0.9
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None


class OpenAIChatRequest(BaseModel):
    model: str = "metis-3.0"
    messages: List[Dict[str, str]] = []
    max_tokens: int = 200
    temperature: float = 0.8
    top_p: float = 0.9
    stream: bool = False


# ── Server Setup ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    ckpt_dir = os.environ.get("METIS_CHECKPOINT_DIR", "checkpoints")
    device = os.environ.get("METIS_DEVICE", None)
    logger.info(f"Loading model from {ckpt_dir}...")
    try:
        model, tokenizer, config = load_model_and_tokenizer(ckpt_dir, device)
        model_globals["model"] = model
        model_globals["tokenizer"] = tokenizer
        model_globals["config"] = config
        logger.info(f"Model loaded: {config.n_params} params, device={config.device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Start with: METIS_CHECKPOINT_DIR=path/to/checkpoints metis serve")
    yield


app = FastAPI(
    title="Μῆτις (Metis) API",
    description="REST API for the Metis language model",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Generate Helper ───────────────────────────────────────────────────────────

def _generate(prompt: str, req: GenerateRequest) -> str:
    """Internal generation call."""
    model = model_globals["model"]
    tokenizer = model_globals["tokenizer"]
    config = model_globals["config"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return generate_text(
        model, tokenizer, prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        device=config.device,
        stop_string=req.stop,
    )


async def _stream_generator(prompt: str, req: GenerateRequest):
    """Async generator for streaming generation with real-time token delivery."""
    model = model_globals["model"]
    tokenizer = model_globals["tokenizer"]
    config = model_globals["config"]
    if model is None:
        yield f"data: {json.dumps({'error': 'Model not loaded'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Use a queue to receive tokens from the generation thread
    token_queue: Queue = Queue()
    done_event = threading.Event()

    def stream_token(token: str):
        """Called from the generation thread for each token."""
        token_queue.put(("token", token))

    def generate_in_thread():
        """Run generation in a separate thread, streaming tokens."""
        try:
            generate_text(
                model, tokenizer, prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                device=config.device,
                stop_string=req.stop,
                stream_callback=stream_token,
            )
        finally:
            done_event.set()

    # Start generation in background thread
    thread = threading.Thread(target=generate_in_thread, daemon=True)
    thread.start()

    # Yield tokens as they arrive from the queue
    while not done_event.is_set() or not token_queue.empty():
        try:
            event_type, data = token_queue.get(timeout=0.1)
            yield f"data: {json.dumps({'token': data})}\n\n"
        except Empty:
            continue

    # Send final done signal
    yield "data: [DONE]\n\n"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "Μῆτις (Metis)",
        "version": "3.0",
        "docs": "/docs",
        "endpoints": ["/generate", "/chat", "/stream", "/info",
                      "/v1/completions", "/v1/chat/completions"],
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate text from a prompt."""
    if model_globals["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded (train one first)")

    t0 = time.time()
    text = _generate(req.prompt, req)
    elapsed = (time.time() - t0) * 1000
    tokenizer = model_globals["tokenizer"]

    return GenerateResponse(
        text=text,
        tokens_generated=len(tokenizer.encode(text)) if tokenizer else 0,
        generation_time_ms=round(elapsed, 2),
    )


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """Chat with the model using message history."""
    if model_globals["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Format messages into prompt
    prompt = ""
    for msg in req.messages:
        if msg.role == "system":
            prompt += f"System: {msg.content}\n"
        elif msg.role == "user":
            prompt += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"Metis: {msg.content}\n"
    prompt += "Metis:"

    gen_req = GenerateRequest(
        prompt=prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        stream=req.stream,
    )

    if req.stream:
        return StreamingResponse(
            _stream_generator(prompt, gen_req),
            media_type="text/event-stream",
        )

    text = _generate(prompt, gen_req)
    response_text = text[len(prompt):].strip().split("\n")[0]
    return {"response": response_text, "role": "assistant"}


@app.get("/info", response_model=ModelInfo)
async def info():
    """Get model metadata."""
    config = model_globals["config"]
    tokenizer = model_globals["tokenizer"]
    if config is None:
        return ModelInfo(
            parameters="(not loaded)",
            tokenizer="(not loaded)",
            device="(not loaded)",
            max_seq_len=0,
            vocab_size=0,
            dtype="(not loaded)",
        )

    return ModelInfo(
        parameters=config.n_params,
        tokenizer=getattr(tokenizer, "encoding_name", "char"),
        device=config.device,
        max_seq_len=config.max_seq_len,
        vocab_size=config.vocab_size,
        dtype=str(next(model_globals["model"].parameters()).dtype) if model_globals["model"] else "unknown",
    )


# ── OpenAI-compatible endpoints ───────────────────────────────────────────────

@app.post("/v1/completions")
async def openai_completions(req: OpenAICompletionRequest):
    """OpenAI-compatible completions endpoint."""
    prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
    gen_req = GenerateRequest(
        prompt=prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stream=req.stream,
        stop=req.stop[0] if isinstance(req.stop, list) else req.stop,
    )

    t0 = time.time()
    text = _generate(prompt, gen_req) if model_globals["model"] else ""
    elapsed = time.time() - t0

    response = {
        "id": f"cmpl-{int(time.time())}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "text": text[len(prompt):] if text.startswith(prompt) else text,
            "index": 0,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(model_globals["tokenizer"].encode(prompt)) if model_globals["tokenizer"] else 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    return response


@app.post("/v1/chat/completions")
async def openai_chat(req: OpenAIChatRequest):
    """OpenAI-compatible chat completions endpoint."""
    # Format messages
    prompt = ""
    for msg in req.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt += f"System: {content}\n"
        elif role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Metis: {content}\n"
    prompt += "Metis:"

    gen_req = GenerateRequest(
        prompt=prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stream=req.stream,
    )

    t0 = time.time()
    text = _generate(prompt, gen_req) if model_globals["model"] else ""
    elapsed = time.time() - t0
    response_text = text[len(prompt):].strip().split("\n")[0]

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(model_globals["tokenizer"].encode(prompt)),
            "completion_tokens": len(model_globals["tokenizer"].encode(response_text)),
            "total_tokens": len(model_globals["tokenizer"].encode(prompt + response_text)),
        },
    }


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok" if model_globals["model"] is not None else "no_model",
        "model_loaded": model_globals["model"] is not None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    """Start the API server."""
    import argparse
    parser = argparse.ArgumentParser(description="Metis API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    args = parser.parse_args()

    os.environ["METIS_CHECKPOINT_DIR"] = args.checkpoint_dir
    if args.device:
        os.environ["METIS_DEVICE"] = args.device

    print(f"\n  Μῆτις API Server v3.0")
    print(f"  Docs:  http://{args.host}:{args.port}/docs")
    print(f"  Info:  http://{args.host}:{args.port}/info")
    print(f"  Chat:  POST http://{args.host}:{args.port}/chat")
    print()

    uvicorn.run(
        "metis.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
