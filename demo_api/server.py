"""
Demo backend for FE on :3000 calling :8000.

Run from repository root:
  pip install -r demo_api/requirements-demo-api.txt
  python -m uvicorn demo_api.server:app --host 127.0.0.1 --port 8000

Env:
  DEMO_API_CORS_ORIGINS  comma-separated origins (default: localhost + 127.0.0.1 :3000)
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Dict, List

logger = logging.getLogger("demo_api")

try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from starlette.responses import Response
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Install demo API deps: pip install -r demo_api/requirements-demo-api.txt"
    ) from e


def _cors_origins() -> List[str]:
    raw = os.environ.get(
        "DEMO_API_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(title="Tool-R0 Demo API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    path = request.url.path
    method = request.method
    client = request.client.host if request.client else "?"
    logger.info("request start %s %s from %s", method, path, client)
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request failed %s %s", method, path)
        raise
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "request done %s %s -> %s in %.1fms",
        method,
        path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.on_event("startup")
def on_startup():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    routes = [getattr(r, "path", str(r)) for r in app.routes]
    logger.info("Demo API listening. Registered paths: %s", sorted(set(routes)))


@app.get("/api/health")
def health():
    return {"ok": True, "service": "tool-r0-demo-api"}


@app.post("/api/demo/start")
async def demo_start(request: Request):
    """
    Starts a demo session. FE sends JSON body (optional); response includes session_id
    for follow-up calls if you extend the API later.
    """
    try:
        raw_body = await request.json()
    except Exception:
        raw_body = {}
    session_id = f"demo-{uuid.uuid4().hex[:12]}"
    logger.info("demo_start session_id=%s body_keys=%s", session_id, list(raw_body.keys()) if isinstance(raw_body, dict) else type(raw_body))
    return JSONResponse(
        {
            "ok": True,
            "session_id": session_id,
            "sessionId": session_id,
            "message": "Demo session started",
            "received": raw_body if isinstance(raw_body, dict) else {},
        }
    )


@app.options("/api/demo/start")
async def demo_start_options():
    """Explicit OPTIONS so preflight never falls through to 404."""
    return Response(status_code=204)
