"""
Sprint 2 — VLM Analysis (The Brain)

Sends the best keyframes to a Vision-Language Model and gets back
the game's DNA as structured JSON.

Two backends supported (configured in config.yaml):
  - "moondream"  → Free, local via Ollama. Fits in 4GB VRAM. Default.
  - "openai"     → GPT-4o mini via API. Better quality, ~$0.01/trailer.

Architecture doc reference:
  Use a Vision-Language Model (VLM) to analyze keyframes to extract
  color palettes, character shapes, and game genre.
"""
import os
import json
import base64
from pathlib import Path

from pipeline.validator import (
    GameDNA, validate_vlm_output, get_safe_defaults, build_retry_prompt
)

MAX_RETRIES = 3

# ── The system prompt sent to the VLM ─────────────────────────────────────────
# This is the most important prompt in the whole project.
# The VLM's output drives everything downstream.
VLM_SYSTEM_PROMPT = """You are a video game analyst and pixel art director.
You will be shown keyframes from a modern game trailer.
Your job is to extract the game's DNA for an 8-bit NES-style demake.

You MUST respond with ONLY valid JSON. No explanation, no markdown, no preamble.
Raw JSON only, starting with { and ending with }.

Required schema:
{
  "title_guess": "your best guess at the game title",
  "genre": "MUST be one of: wave_shooter | top_down_action_rpg | open_world_sandbox | side_scroll_platformer",
  "setting": "one sentence describing the world (e.g. post-apocalyptic city, fantasy dungeon)",
  "color_palette": ["#hex1", "#hex2", "#hex3", "#hex4"],
  "player_description": "describe the main character for pixel art generation, mention view angle",
  "enemy_description": "describe the main enemy type for pixel art generation",
  "boss_description": "describe the boss or main antagonist for pixel art generation",
  "environment_description": "describe the environment tiles and background for pixel art",
  "music_vibe": "MUST be one of: intense_action | dark_horror | epic_adventure | urban_gritty | mysterious",
  "music_tempo": "MUST be one of: slow | medium | fast | frantic",
  "confidence": 0.0
}

Genre selection guide:
- wave_shooter: survival/horde modes, shooting enemies in waves (CoD Zombies, Vampire Survivors)
- top_down_action_rpg: combat with HP/MP bars, exploration (Zelda, Kingdom Hearts, Dark Souls)
- open_world_sandbox: free roaming, missions, vehicles (GTA, Saints Row, Cyberpunk)
- side_scroll_platformer: jumping on platforms, left-to-right (Mario, Sonic, Hollow Knight)

Set confidence between 0.0 (very unsure) and 1.0 (very sure).
If you cannot identify the game clearly, pick the closest genre and set confidence low."""


def run_vlm_analysis(best_frames: list[str], output_dir: str, config: dict) -> GameDNA:
    """
    Analyze the best keyframes with a VLM and return validated GameDNA.

    Args:
        best_frames: List of paths to the top-scored PNG keyframes
        output_dir:  /outputs/{demake_id}/ — game_dna.json written here
        config:      Loaded config.yaml dict

    Returns:
        Validated GameDNA object (never raises — falls back to defaults)
    """
    pipeline_cfg = config.get("pipeline", {})
    backend      = pipeline_cfg.get("vlm_backend", "moondream")

    print(f"[VLM] Using backend: {backend} | Frames: {len(best_frames)}")

    # Try to get real VLM output
    dna = None
    try:
        if backend == "openai":
            api_key = pipeline_cfg.get("openai_api_key", "")
            if not api_key:
                print("[VLM] OpenAI key missing — falling back to moondream")
                dna = _run_moondream(best_frames)
            else:
                dna = _run_openai(best_frames, api_key)
        else:
            dna = _run_moondream(best_frames)
    except Exception as e:
        print(f"[VLM] Backend error: {e} — using safe defaults")
        dna = None

    # If VLM completely failed, use safe defaults
    if dna is None:
        print("[VLM] All attempts failed — using safe defaults")
        dna = get_safe_defaults("wave_shooter")

    # Write game_dna.json to output directory
    dna_path = os.path.join(output_dir, "game_dna.json")
    with open(dna_path, "w") as f:
        json.dump(dna.model_dump(), f, indent=2)
    print(f"[VLM] game_dna.json written: genre={dna.genre}, confidence={dna.confidence:.2f}")

    return dna


# ── Moondream backend (free, local, Ollama) ────────────────────────────────────
def _run_moondream(best_frames: list[str]) -> GameDNA | None:
    """
    Run Moondream via Ollama for free local VLM inference.

    Requires: ollama running locally with moondream pulled:
        ollama pull moondream
        ollama serve
    """
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests not installed — run: pip install requests")

    ollama_url = "http://localhost:11434/api/chat"

    # Build message with up to 3 frames (Moondream handles multiple images)
    frames_to_use = best_frames[:3]
    images_b64 = []
    for frame_path in frames_to_use:
        with open(frame_path, "rb") as f:
            images_b64.append(base64.b64encode(f.read()).decode())

    messages = [
        {
            "role": "user",
            "content": VLM_SYSTEM_PROMPT,
            "images": images_b64
        }
    ]

    last_output = ""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                ollama_url,
                json={
                    "model":  "moondream",
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,   # Low temp = more consistent JSON
                        "num_predict": 600,
                    }
                },
                timeout=120  # Moondream can be slow on first run
            )
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]
            last_output = raw

            dna = validate_vlm_output(raw, attempt)
            if dna:
                return dna

            # Failed validation — add correction to conversation
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": build_retry_prompt(raw, "Schema validation failed")
            })

        except requests.exceptions.ConnectionError:
            print("[VLM] Ollama not running. Start it with: ollama serve")
            print("[VLM] Then pull moondream: ollama pull moondream")
            return None
        except Exception as e:
            print(f"[VLM] Moondream attempt {attempt + 1} error: {e}")

    print("[VLM] Moondream: all retries exhausted")
    return None


# ── OpenAI GPT-4o mini backend (optional, cloud) ──────────────────────────────
def _run_openai(best_frames: list[str], api_key: str) -> GameDNA | None:
    """
    Run GPT-4o mini for higher quality VLM analysis.
    Only used if vlm_backend: "openai" in config.yaml and api_key is set.
    Cost: ~$0.01 per trailer.
    """
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests not installed")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Build image content blocks (up to 5 frames for GPT-4o)
    frames_to_use = best_frames[:5]
    content = [{"type": "text", "text": VLM_SYSTEM_PROMPT}]

    for frame_path in frames_to_use:
        with open(frame_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content.append({
            "type": "image_url",
            "image_url": {
                "url":    f"data:image/png;base64,{b64}",
                "detail": "low"  # Low detail = cheaper, still enough for game analysis
            }
        })

    messages = [{"role": "user", "content": content}]
    last_output = ""

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json={
                    "model":       "gpt-4o-mini",
                    "messages":    messages,
                    "max_tokens":  600,
                    "temperature": 0.1,
                },
                timeout=60
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            last_output = raw

            dna = validate_vlm_output(raw, attempt)
            if dna:
                return dna

            # Retry with correction
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": build_retry_prompt(raw, "Schema validation failed")
            })

        except Exception as e:
            print(f"[VLM] OpenAI attempt {attempt + 1} error: {e}")

    print("[VLM] OpenAI: all retries exhausted")
    return None


# ── Genre → template mapping ───────────────────────────────────────────────────
def match_genre_template(dna: GameDNA, config: dict) -> str:
    """
    Maps the VLM's genre string to a template file.
    Falls back to config default if confidence is low.

    Returns the template_id string (e.g. "wave_shooter").
    """
    fallback = config.get("genre_templates", {}).get("default_fallback", "wave_shooter")

    # Low confidence → use fallback
    if dna.confidence < 0.4:
        print(f"[Genre] Low confidence ({dna.confidence:.2f}) — using fallback: {fallback}")
        return fallback

    print(f"[Genre] Matched template: {dna.genre} (confidence: {dna.confidence:.2f})")
    return dna.genre