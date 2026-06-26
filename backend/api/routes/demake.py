"""
Core Demake API routes — matches the architecture doc exactly.

POST /api/v1/demake/upload       — upload video, create DB record, enqueue job
GET  /api/v1/demake/{id}/status  — poll pipeline progress
GET  /api/v1/demake/{id}/manifest — fetch completed game manifest
GET  /api/v1/demake/{id}/asset/{filename} — serve generated files
WS   /ws/demake/{id}             — real-time progress stream
"""
import os
import json
import uuid
import aiofiles
from datetime import datetime

from fastapi import (
    APIRouter, Depends, File, UploadFile,
    HTTPException, WebSocket, WebSocketDisconnect
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from database.db import get_db
from database.models import Demake
from pipeline.orchestrator import enqueue, register_ws, unregister_ws

router = APIRouter()

# ── Config ─────────────────────────────────────────────────────────────────────
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100MB

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Maps status string → (stage number, progress pct, human message)
STATUS_META = {
    "queued":             (1, 5,   "Waiting in queue..."),
    "extracting_frames":  (2, 15,  "Extracting keyframes from video..."),
    "analyzing":          (3, 30,  "Analyzing game DNA with vision model..."),
    "matching_genre":     (4, 45,  "Matching genre template..."),
    "generating_sprites": (5, 60,  "Generating pixel art sprites..."),
    "generating_audio":   (6, 80,  "Composing chiptune music..."),
    "assembling":         (7, 92,  "Assembling game manifest..."),
    "ready":              (8, 100, "Your demake is ready to play!"),
    "failed":             (0, 0,   "Pipeline failed."),
}


# ── POST /api/v1/demake/upload ─────────────────────────────────────────────────
@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    db:   Session    = Depends(get_db)
):
    """
    Accept an MP4 upload, save it to disk, create a DB record, enqueue the job.

    Returns: { demake_id, status }
    Errors:  413 if too large | 415 if not MP4
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(
            status_code=415,
            detail="Only MP4 files are accepted. Please upload a game trailer in MP4 format."
        )

    # Read file and check size
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is 100MB, got {len(contents) / 1024 / 1024:.1f}MB."
        )

    # Generate demake ID and save file
    demake_id  = str(uuid.uuid4())
    safe_name  = f"{demake_id}.mp4"
    save_path  = os.path.join(UPLOAD_DIR, safe_name)

    async with aiofiles.open(save_path, "wb") as f:
        await f.write(contents)

    # Create output directory for this run
    run_output = os.path.join(OUTPUT_DIR, demake_id)
    os.makedirs(run_output, exist_ok=True)

    # Write to DB
    demake = Demake(
        id          = demake_id,
        title       = file.filename.replace(".mp4", "").replace("_", " ").title(),
        status      = "queued",
        source_path = save_path,
        created_at  = datetime.utcnow(),
    )
    db.add(demake)
    db.commit()
    db.refresh(demake)

    # Enqueue the job — pipeline worker picks it up immediately
    await enqueue(demake_id)

    print(f"[Upload] {file.filename} → {demake_id[:8]} | {len(contents) / 1024:.1f} KB")

    return {
        "demake_id": demake_id,
        "status":    "queued",
        "message":   "Video uploaded successfully. Pipeline is starting..."
    }


# ── GET /api/v1/demake/{id}/status ────────────────────────────────────────────
@router.get("/{demake_id}/status")
def get_status(demake_id: str, db: Session = Depends(get_db)):
    """
    Poll the current pipeline stage for a demake.

    Returns: { demake_id, status, stage, total_stages, progress_pct, message }
    """
    demake = db.query(Demake).filter_by(id=demake_id).first()
    if not demake:
        raise HTTPException(
            status_code=404,
            detail=f"Demake '{demake_id}' not found."
        )

    stage, pct, msg = STATUS_META.get(demake.status, (0, 0, demake.status))

    response = {
        "demake_id":    demake.id,
        "status":       demake.status,
        "stage":        stage,
        "total_stages": 8,
        "progress_pct": pct,
        "message":      msg,
    }

    if demake.status == "failed":
        response["error"] = demake.error_message

    if demake.status == "ready" and demake.completed_at:
        response["completed_at"] = demake.completed_at.isoformat()

    return response


# ── GET /api/v1/demake/{id}/manifest ──────────────────────────────────────────
@router.get("/{demake_id}/manifest")
def get_manifest(demake_id: str, db: Session = Depends(get_db)):
    """
    Fetch the completed game manifest JSON once status = "ready".
    This is what the Phaser frontend loads to boot the game.

    Returns: Full manifest JSON
    Errors:  404 if not found | 409 if pipeline not complete yet
    """
    demake = db.query(Demake).filter_by(id=demake_id).first()
    if not demake:
        raise HTTPException(status_code=404, detail="Demake not found.")

    if demake.status == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {demake.error_message}"
        )

    if demake.status != "ready":
        stage, pct, msg = STATUS_META.get(demake.status, (0, 0, "Processing..."))
        raise HTTPException(
            status_code=409,
            detail={
                "message":      "Demake is still being generated.",
                "status":       demake.status,
                "progress_pct": pct,
                "stage":        stage,
            }
        )

    manifest_path = os.path.join(OUTPUT_DIR, demake_id, "manifest.json")
    if not os.path.exists(manifest_path):
        # Sprint 1: return a stub manifest so frontend has something to load
        return _stub_manifest(demake_id, demake.title or "UNTITLED DEMAKE")

    with open(manifest_path) as f:
        return json.load(f)


# ── GET /api/v1/demake/{id}/asset/{filename} ──────────────────────────────────
@router.get("/{demake_id}/asset/{filename}")
def get_asset(demake_id: str, filename: str, db: Session = Depends(get_db)):
    """
    Serve a generated asset file (sprite PNG, MIDI, etc.).
    Checks the demake exists before serving to prevent directory traversal.
    """
    demake = db.query(Demake).filter_by(id=demake_id).first()
    if not demake:
        raise HTTPException(status_code=404, detail="Demake not found.")

    # Security: strip any path components from filename
    safe_filename = os.path.basename(filename)
    asset_path    = os.path.join(OUTPUT_DIR, demake_id, safe_filename)

    if not os.path.exists(asset_path):
        raise HTTPException(
            status_code=404,
            detail=f"Asset '{safe_filename}' not found for this demake."
        )

    return FileResponse(asset_path)


# ── WebSocket /ws/demake/{id} ──────────────────────────────────────────────────
@router.websocket("/ws/{demake_id}")
async def websocket_status(websocket: WebSocket, demake_id: str):
    """
    Real-time pipeline status stream.
    Client connects, receives updates as the pipeline progresses.
    Connection closes automatically when the job finishes or fails.
    """
    await websocket.accept()
    register_ws(demake_id, websocket)

    try:
        # Send current status immediately on connect
        db = next(get_db())
        demake = db.query(Demake).filter_by(id=demake_id).first()
        if demake:
            stage, pct, msg = STATUS_META.get(demake.status, (0, 0, demake.status))
            await websocket.send_json({
                "demake_id":    demake_id,
                "status":       demake.status,
                "stage":        stage,
                "total_stages": 8,
                "progress_pct": pct,
                "message":      msg,
            })
        db.close()

        # Keep connection alive — pipeline broadcasts updates via register_ws
        while True:
            await websocket.receive_text()   # Client can send "ping" to keep alive

    except WebSocketDisconnect:
        pass
    finally:
        unregister_ws(demake_id, websocket)


# ── Stub manifest for Sprint 1 ────────────────────────────────────────────────
def _stub_manifest(demake_id: str, title: str) -> dict:
    """
    Returns a hardcoded Wave Shooter manifest for Sprint 1 testing.
    Sprints 2–4 replace this with the real generated manifest.
    """
    return {
        "demake_id": demake_id,
        "title":     title.upper(),
        "template":  "wave_shooter",
        "note":      "STUB MANIFEST — Sprint 1. Real generation coming in Sprint 3.",
        "palette": {
            "primary":    "#1a0a00",
            "secondary":  "#3d2b1f",
            "accent":     "#8b1a00",
            "highlight":  "#ff4400",
            "background": "#0d0500",
        },
        "game_config": {
            "physics":  {"gravity": 0, "player_speed": 130, "bullet_speed": 380},
            "waves":    {"start_count": 6, "multiplier": 1.4, "boss_every": 5},
            "player":   {"max_hp": 100, "max_ammo": 30, "reload_time_ms": 1800, "fire_rate_ms": 150},
            "enemy":    {"base_hp": 30,  "base_speed": 55, "damage": 10, "hp_scale": 1.2, "speed_scale": 1.05},
            "boss":     {"hp_mult": 8,   "speed_mult": 0.6, "damage_mult": 3, "size_mult": 2.0}
        },
        "assets": {
            "sprites": {},
            "audio":   {}
        }
    }