"""
Pipeline Orchestrator — the heart of the backend.

Manages an asyncio queue of demake jobs. Each job runs through
all pipeline stages in order, updating the DB status at each step.

Sprint 1: All stages are STUBS — they just log and sleep.
Sprint 2: ingestion + vlm_analysis get real implementations.
Sprint 3: sprite_gen + audio_gen get real implementations.
Sprint 4: Everything is wired together.
"""
import os
import asyncio
import traceback
from datetime import datetime
from sqlalchemy.orm import Session

from database.db import SessionLocal
from database.models import Demake

# Sprint 2 — real implementations wired in
from pipeline.ingestion    import run_ingestion
from pipeline.vlm_analysis import run_vlm_analysis, match_genre_template
from pipeline.validator    import GameDNA
import yaml as _yaml

def _load_config():
    try:
        with open("config.yaml") as f:
            return _yaml.safe_load(f)
    except Exception:
        return {}
_CONFIG = _load_config()

# ── Global job queue ──────────────────────────────────────────────────────────
# asyncio.Queue is thread-safe and works perfectly for a single-worker pipeline.
# If we ever need parallel jobs we can swap this for Celery (see architecture doc).
_job_queue: asyncio.Queue = asyncio.Queue()

# Active WebSocket connections keyed by demake_id
# Format: { demake_id: [websocket, ...] }
_ws_connections: dict[str, list] = {}


def register_ws(demake_id: str, ws):
    """Register a WebSocket connection to receive pipeline updates."""
    if demake_id not in _ws_connections:
        _ws_connections[demake_id] = []
    _ws_connections[demake_id].append(ws)


def unregister_ws(demake_id: str, ws):
    if demake_id in _ws_connections:
        _ws_connections[demake_id].discard(ws) if hasattr(
            _ws_connections[demake_id], 'discard'
        ) else None
        try:
            _ws_connections[demake_id].remove(ws)
        except ValueError:
            pass


async def _broadcast(demake_id: str, payload: dict):
    """Send a status update to all WebSocket clients watching this demake."""
    dead = []
    for ws in _ws_connections.get(demake_id, []):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        unregister_ws(demake_id, ws)


# ── Stage definitions ─────────────────────────────────────────────────────────
# Each stage is a tuple of:
#   (status_label, stage_number, progress_pct, human_readable_message)
STAGES = [
    ("extracting_frames",  2, 15,  "Extracting keyframes from video..."),
    ("analyzing",          3, 30,  "Analyzing game DNA with vision model..."),
    ("matching_genre",     4, 45,  "Matching genre template..."),
    ("generating_sprites", 5, 60,  "Generating pixel art sprites..."),
    ("generating_audio",   6, 80,  "Composing chiptune music..."),
    ("assembling",         7, 92,  "Assembling game manifest..."),
]


async def _set_status(db: Session, demake: Demake, status: str,
                      stage: int, pct: int, msg: str):
    """Update DB status and broadcast to WebSocket watchers."""
    demake.status = status
    db.commit()
    payload = {
        "demake_id": demake.id,
        "status":    status,
        "stage":     stage,
        "total_stages": 8,
        "progress_pct": pct,
        "message":   msg,
    }
    await _broadcast(demake.id, payload)
    print(f"[Pipeline] [{demake.id[:8]}] Stage {stage}/8 ({pct}%) — {msg}")


async def _run_pipeline(demake_id: str):
    """
    Run the full generation pipeline for a single demake.

    Sprint 1: All stages are stubs (sleep + log).
    Real implementations slot in during Sprints 2 & 3.
    """
    db = SessionLocal()
    try:
        demake = db.query(Demake).filter_by(id=demake_id).first()
        if not demake:
            print(f"[Pipeline] ERROR — demake {demake_id} not found in DB")
            return

        output_dir = os.path.join("outputs", demake_id)
        os.makedirs(output_dir, exist_ok=True)

        # ── Stage 2: Extract frames (REAL — Sprint 2) ─────────────────────
        await _set_status(db, demake, "extracting_frames", 2, 15,
                          "Extracting keyframes from video...")
        ingestion_result = await asyncio.get_event_loop().run_in_executor(
            None, run_ingestion, demake.source_path, output_dir
        )
        best_frames = ingestion_result["best_frames"]
        print(f"[Pipeline] Extracted {ingestion_result['frame_count']} frames, "
              f"best {len(best_frames)} selected")

        # ── Stage 3: VLM Analysis (REAL — Sprint 2) ───────────────────────
        await _set_status(db, demake, "analyzing", 3, 30,
                          "Analyzing game DNA with vision model...")
        dna = await asyncio.get_event_loop().run_in_executor(
            None, run_vlm_analysis, best_frames, output_dir, _CONFIG
        )

        # ── Stage 4: Genre Matching + Validation (REAL — Sprint 2) ───────
        await _set_status(db, demake, "matching_genre", 4, 45,
                          "Matching genre template...")
        template_id = match_genre_template(dna, _CONFIG)

        # Update DB with extracted config
        import json as _json
        from database.models import GameConfig
        game_cfg = GameConfig(
            demake_id     = demake_id,
            template_id   = template_id,
            genre         = dna.genre,
            color_palette = _json.dumps(dna.color_palette),
            mechanics     = _json.dumps({"music_vibe": dna.music_vibe,
                                         "music_tempo": dna.music_tempo}),
            vlm_raw_output = dna.model_dump_json(),
        )
        db.add(game_cfg)
        db.commit()
        print(f"[Pipeline] Genre matched: {template_id}")

        # ── Stage 5: Sprite Generation (STUB — Sprint 3) ──────────────────
        await _set_status(db, demake, "generating_sprites", 5, 60,
                          "Generating pixel art sprites...")
        await asyncio.sleep(1.0)   # STUB — Sprint 3

        # ── Stage 6: Audio Generation (STUB — Sprint 3) ───────────────────
        await _set_status(db, demake, "generating_audio", 6, 80,
                          "Composing chiptune music...")
        await asyncio.sleep(0.5)   # STUB — Sprint 3

        # ── Stage 7: Manifest Assembly (STUB — Sprint 4) ──────────────────
        await _set_status(db, demake, "assembling", 7, 92,
                          "Assembling game manifest...")
        await asyncio.sleep(0.3)   # STUB — Sprint 4

        # ── Stage 8: Done ─────────────────────────────────────────────────
        demake.status       = "ready"
        demake.completed_at = datetime.utcnow()
        db.commit()

        await _broadcast(demake.id, {
            "demake_id":    demake.id,
            "status":       "ready",
            "stage":        8,
            "total_stages": 8,
            "progress_pct": 100,
            "message":      "Your demake is ready to play!",
        })
        print(f"[Pipeline] [{demake_id[:8]}] ✓ Complete")

    except Exception as e:
        # Never crash the worker — mark as failed and keep going
        print(f"[Pipeline] [{demake_id[:8]}] FAILED: {e}")
        traceback.print_exc()
        try:
            demake = db.query(Demake).filter_by(id=demake_id).first()
            if demake:
                demake.status        = "failed"
                demake.error_message = str(e)
                db.commit()
                await _broadcast(demake_id, {
                    "demake_id": demake_id,
                    "status":    "failed",
                    "progress_pct": 0,
                    "message":   f"Pipeline failed: {e}",
                })
        except Exception as inner:
            print(f"[Pipeline] Could not write failure to DB: {inner}")
    finally:
        db.close()


async def _worker():
    """
    Single background worker — pulls jobs from the queue one at a time.
    Runs forever as an asyncio task started at server startup.
    """
    print("[Worker] Pipeline worker started — waiting for jobs")
    while True:
        demake_id = await _job_queue.get()
        print(f"[Worker] Starting job: {demake_id[:8]}...")
        await _run_pipeline(demake_id)
        _job_queue.task_done()


async def enqueue(demake_id: str):
    """Add a demake job to the processing queue."""
    await _job_queue.put(demake_id)
    print(f"[Queue] Enqueued job {demake_id[:8]} (queue size: {_job_queue.qsize()})")


def start_worker(app):
    """
    Called at FastAPI startup — creates the background worker task.
    The worker runs for the lifetime of the server process.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    loop.create_task(_worker())
    print("[Worker] Background pipeline worker registered")