"""
Demake Engine — FastAPI Backend
Entry point. Run with: uvicorn main:app --reload

Architecture doc reference:
  Backend Framework: Python with FastAPI (async processing + WebSocket updates)
  Database: SQLite with WAL mode
  Job Queue: asyncio queue (single worker, sequential jobs)
"""
import os
import yaml
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from database.db import init_db
from pipeline.orchestrator import start_worker
from api.routes import demake as demake_routes
from api.routes import health as health_routes


# ── Load config ───────────────────────────────────────────────────────────────
def load_config(path: str = "config.yaml") -> dict:
    if os.path.exists(path):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        print(f"[Config] Loaded from {path}")
        return cfg
    print(f"[Config] {path} not found — using defaults")
    return {}

config = load_config()

# Expose config values as env vars for routes to read
OUTPUT_DIR = config.get("storage", {}).get("output_dir", "outputs")
UPLOAD_DIR = config.get("storage", {}).get("upload_dir", "uploads")
os.environ["OUTPUT_DIR"] = OUTPUT_DIR
os.environ["UPLOAD_DIR"] = UPLOAD_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── App lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on startup and shutdown."""
    print("=" * 50)
    print("  DEMAKE ENGINE — Starting up")
    print("=" * 50)

    # Initialize DB (creates tables if they don't exist)
    init_db()

    # Start the background pipeline worker
    start_worker(app)

    yield  # Server is running

    print("[Shutdown] Demake Engine shutting down...")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Demake Engine API",
    description="Turn any game trailer into a playable 8-bit browser demake.",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow the frontend (running on a different port during dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(health_routes.router, tags=["Health"])
app.include_router(
    demake_routes.router,
    prefix="/api/v1/demake",
    tags=["Demake"]
)
# WebSocket is mounted separately (no /api/v1 prefix needed)
app.add_api_websocket_route(
    "/ws/demake/{demake_id}",
    demake_routes.websocket_status
)

# Serve generated outputs as static files
# e.g. /files/abc123/sprite_player.png
app.mount(
    "/files",
    StaticFiles(directory=OUTPUT_DIR),
    name="outputs"
)


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service":  "Demake Engine API",
        "version":  "0.1.0",
        "status":   "running",
        "docs":     "/docs",
        "endpoints": {
            "upload":   "POST /api/v1/demake/upload",
            "status":   "GET  /api/v1/demake/{id}/status",
            "manifest": "GET  /api/v1/demake/{id}/manifest",
            "asset":    "GET  /api/v1/demake/{id}/asset/{filename}",
            "ws":       "WS   /ws/demake/{id}",
            "health":   "GET  /health",
        }
    }