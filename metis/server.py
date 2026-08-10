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

import json
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from queue import Empty, Queue

from pydantic import BaseModel, Field

try:
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.concurrency import run_in_threadpool
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
except ImportError:
    print("❌ fastapi not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)

from metis import (
    generate_text,
    load_model_and_tokenizer,
)

logger = logging.getLogger("metis.server")

# ── Global model state ───────────────────────────────────────────────────────
model_globals = {"model": None, "tokenizer": None, "config": None}


# ── Request / Response Models ─────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt to generate from")
    max_tokens: int = Field(200, ge=1, le=4096, description="Max tokens to generate")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: int | None = Field(40, ge=0, description="Top-k filtering")
    top_p: float | None = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling")
    repetition_penalty: float = Field(1.1, ge=1.0, le=5.0, description="Repetition penalty")
    stream: bool = Field(False, description="Stream tokens via SSE")
    stop: str | None = Field(None, description="Stop string")


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: user/assistant/system")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., description="Chat messages")
    max_tokens: int = Field(200, ge=1, le=4096)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int | None = Field(40)
    top_p: float | None = Field(0.9)
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
    prompt: str | list[str] = ""
    max_tokens: int = Field(200, ge=1, le=4096,
                            description="Max tokens to generate")
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    stream: bool = False
    stop: str | list[str] | None = None


class OpenAIChatRequest(BaseModel):
    model: str = "metis-3.0"
    messages: list[dict[str, str]] = []
    max_tokens: int = Field(200, ge=1, le=4096,
                            description="Max tokens to generate")
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    stream: bool = False


# ── Server Setup ──────────────────────────────────────────────────────────────

async def _require_auth(authorization: str | None = Header(default=None)) -> None:
    """Optional bearer-token auth for the generation endpoints.

    The server binds 0.0.0.0 by default, so without auth anyone who can reach
    the port can burn GPU/API tokens. When ``METIS_API_KEY`` is set, requests
    must send ``Authorization: Bearer <key>`` (401 otherwise). When unset the
    server stays open — intended for local / trusted-network usage — so
    enabling auth never breaks an existing deployment.
    """
    api_key = os.environ.get("METIS_API_KEY")
    if not api_key:
        return
    if authorization != f"Bearer {api_key}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


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
    except SystemExit:
        # load_model_and_tokenizer calls sys.exit(1) when checkpoints are
        # missing (fine for the CLI, but a server must keep running — /health
        # and /info stay up so operators can see the "no_model" state).
        logger.error("Failed to load model: checkpoints/tokenizer not found")
        logger.error("Start with: METIS_CHECKPOINT_DIR=path/to/checkpoints metis serve")
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

# CORS: default to wide-open (any origin), overridable via METIS_CORS_ORIGINS
# as a comma-separated list. ``allow_credentials=True`` is incompatible with a
# wildcard origin (browsers ignore the credential header), so credentials are
# only enabled when an explicit origin list is configured.
_cors_origins = [
    o.strip() for o in os.environ.get("METIS_CORS_ORIGINS", "*").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_origins != ["*"],
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


def _strip_prompt(text: str, prompt: str, tokenizer) -> str:
    """Remove the tokenizer-faithful prefix of ``prompt`` from generated text.

    ``generate_text`` returns ``decode(encode(prompt) + new_ids)``. For a char
    tokenizer that decoded prefix equals ``prompt``, so a plain
    ``text[len(prompt):]`` slice is exact; for BPE tokenizers (cl100k_base,
    o200k_base) Unicode normalization can make the decoded prefix differ from
    the input string and a character-count slice then starts mid-token. Strip
    using the tokenizer's own round-trip so the boundary is token-aligned, and
    fall back to the full text if the prefix does not match (never truncate a
    response into the middle of a word).
    """
    if not prompt or tokenizer is None:
        return text
    prefix = tokenizer.decode(tokenizer.encode(prompt))
    return text[len(prefix):] if text.startswith(prefix) else text


async def _stream_tokens(prompt: str, req: GenerateRequest):
    """Run ``generate_text`` in a daemon thread, yielding ``(type, payload)``.

    Generation is CPU/GPU-bound and must not block the async event loop, so it
    runs in a background thread that pushes tokens into a queue. Any exception
    in the thread is captured and re-raised in this async frame, so a
    mid-generation failure (CUDA OOM, dtype/shape error) surfaces as a visible
    error instead of silently dying in the thread and leaving the client to
    hang waiting for a ``[DONE]`` that never differs from a live stream.
    """
    model = model_globals["model"]
    tokenizer = model_globals["tokenizer"]
    config = model_globals["config"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    token_queue: Queue = Queue()
    done_event = threading.Event()
    failure: dict[str, Exception] = {}

    def stream_token(token: str):
        """Called from the generation thread for each token."""
        token_queue.put(("token", token))

    def generate_in_thread():
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
        except Exception as e:
            failure["exception"] = e
        finally:
            done_event.set()

    thread = threading.Thread(target=generate_in_thread, daemon=True)
    thread.start()

    while not done_event.is_set() or not token_queue.empty():
        try:
            event_type, data = token_queue.get(timeout=0.1)
            yield event_type, data
        except Empty:
            continue

    if "exception" in failure:
        raise failure["exception"]


async def _stream_generator(prompt: str, req: GenerateRequest):
    """SSE generator for the proprietary stream path (used by /generate, /chat)."""
    try:
        async for _, token in _stream_tokens(prompt, req):
            yield f"data: {json.dumps({'token': token})}\n\n"
    except HTTPException as e:
        yield f"data: {json.dumps({'error': e.detail})}\n\n"
    except Exception as e:
        logger.exception("Streaming generation failed")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    yield "data: [DONE]\n\n"


async def _openai_stream_generator(prompt: str, req: GenerateRequest, *,
                                   chat: bool, model_name: str, req_id: str):
    """Yield OpenAI-compatible SSE chunks for /v1/completions or /v1/chat."""
    created = int(time.time())
    obj = "chat.completion.chunk" if chat else "text_completion"
    try:
        async for _, token in _stream_tokens(prompt, req):
            if chat:
                chunk = {"id": req_id, "object": obj, "created": created,
                         "model": model_name,
                         "choices": [{"index": 0, "delta": {"content": token},
                                      "finish_reason": None}]}
            else:
                chunk = {"id": req_id, "object": obj, "created": created,
                         "model": model_name,
                         "choices": [{"index": 0, "text": token,
                                      "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n"
    except Exception as e:
        logger.exception("Streaming generation failed")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        # Always terminate the stream: OpenAI clients wait for the [DONE]
        # sentinel, so a bare return would leave them hanging on the error path.
        yield "data: [DONE]\n\n"
        return

    # Final chunk carries the finish_reason (required by OpenAI clients).
    finish = {"id": req_id, "object": obj, "created": created, "model": model_name,
              "choices": [{"index": 0, "finish_reason": "stop"}]}
    if chat:
        finish["choices"][0]["delta"] = {}
    else:
        finish["choices"][0]["text"] = ""
    yield f"data: {json.dumps(finish)}\n\n"
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
async def generate(req: GenerateRequest, _auth: None = Depends(_require_auth)):
    """Generate text from a prompt."""
    if model_globals["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded (train one first)")

    if req.stream:
        return StreamingResponse(
            _stream_generator(req.prompt, req),
            media_type="text/event-stream",
        )

    t0 = time.time()
    # generate_text is synchronous and CPU/GPU-bound: run it off the event
    # loop so a long generation does not stall every other request (health,
    # info, and concurrent completions).
    text = await run_in_threadpool(_generate, req.prompt, req)
    elapsed = (time.time() - t0) * 1000
    tokenizer = model_globals["tokenizer"]

    return GenerateResponse(
        text=text,
        tokens_generated=len(tokenizer.encode(text)) if tokenizer else 0,
        generation_time_ms=round(elapsed, 2),
    )


@app.post("/chat")
async def chat_endpoint(req: ChatRequest, _auth: None = Depends(_require_auth)):
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

    text = await run_in_threadpool(_generate, prompt, gen_req)
    response_text = _strip_prompt(text, prompt, model_globals["tokenizer"]).strip().split("\n")[0]
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
        dtype=(
            str(next(model_globals["model"].parameters()).dtype)
            if model_globals["model"] else "unknown"
        ),
    )


# ── OpenAI-compatible endpoints ───────────────────────────────────────────────

@app.post("/v1/completions")
async def openai_completions(req: OpenAICompletionRequest,
                             _auth: None = Depends(_require_auth)):
    """OpenAI-compatible completions endpoint."""
    if model_globals["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
    gen_req = GenerateRequest(
        prompt=prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stream=req.stream,
        stop=req.stop[0] if isinstance(req.stop, list) else req.stop,
    )

    req_id = f"cmpl-{int(time.time())}"
    if req.stream:
        return StreamingResponse(
            _openai_stream_generator(prompt, gen_req, chat=False,
                                     model_name=req.model, req_id=req_id),
            media_type="text/event-stream",
        )

    text = await run_in_threadpool(_generate, prompt, gen_req)
    tokenizer = model_globals["tokenizer"]
    generated = _strip_prompt(text, prompt, tokenizer)
    prompt_tokens = len(tokenizer.encode(prompt)) if tokenizer else 0
    completion_tokens = len(tokenizer.encode(generated)) if tokenizer else 0

    return {
        "id": req_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "text": generated,
            "index": 0,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.post("/v1/chat/completions")
async def openai_chat(req: OpenAIChatRequest, _auth: None = Depends(_require_auth)):
    """OpenAI-compatible chat completions endpoint."""
    if model_globals["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
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

    req_id = f"chatcmpl-{int(time.time())}"
    if req.stream:
        return StreamingResponse(
            _openai_stream_generator(prompt, gen_req, chat=True,
                                     model_name=req.model, req_id=req_id),
            media_type="text/event-stream",
        )

    text = await run_in_threadpool(_generate, prompt, gen_req)
    tokenizer = model_globals["tokenizer"]
    response_text = _strip_prompt(text, prompt, tokenizer).strip().split("\n")[0]
    prompt_tokens = len(tokenizer.encode(prompt)) if tokenizer else 0
    completion_tokens = len(tokenizer.encode(response_text)) if tokenizer else 0

    return {
        "id": req_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
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

    import uvicorn

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

    print("\n  Μῆτις API Server v3.0")
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
