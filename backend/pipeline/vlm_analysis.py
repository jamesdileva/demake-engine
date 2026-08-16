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

    # Title-based genre override — if llava correctly identifies the game,
    # trust the known genre over the VLM classification
    TITLE_GENRE_MAP = {
        "kingdom hearts": "top_down_action_rpg",
        "zelda":          "top_down_action_rpg",
        "diablo":         "top_down_action_rpg",
        "dark souls":     "top_down_action_rpg",
        "elden ring":     "top_down_action_rpg",
        "pokemon":        "turn_based_rpg",
        "final fantasy":  "turn_based_rpg",
        "undertale":      "turn_based_rpg",
        "octopath":       "turn_based_rpg",
        "persona":        "turn_based_rpg",
        "gta":            "open_world_sandbox",
        "grand theft":    "open_world_sandbox",
        "cyberpunk":      "open_world_sandbox",
        "saints row":     "open_world_sandbox",
        "mario":          "side_scroll_platformer",
        "sonic":          "side_scroll_platformer",
        "hollow knight":  "side_scroll_platformer",
        "cod zombies":    "wave_shooter",
        "call of duty":   "wave_shooter",
        "halo":           "wave_shooter",
    }
    title_lower = (dna.title_guess or "").lower()
    for title_key, forced_genre in TITLE_GENRE_MAP.items():
        if title_key in title_lower:
            if dna.genre != forced_genre:
                print(f"[VLM] Title override: '{dna.title_guess}' -> {forced_genre} (was {dna.genre})")
                dna.genre = forced_genre
                dna.confidence = max(dna.confidence, 0.75)
            break

    # Write game_dna.json to output directory
    dna_path = os.path.join(output_dir, "game_dna.json")
    with open(dna_path, "w") as f:
        json.dump(dna.model_dump(), f, indent=2)
    print(f"[VLM] game_dna.json written: genre={dna.genre}, confidence={dna.confidence:.2f}")

    return dna


# ── Moondream backend (free, local, Ollama) ────────────────────────────────────
def _extract_json_from_text(text: str) -> str:
    """Extract JSON object from model output that may contain extra text."""
    import re
    text = text.strip()
    if text.startswith("{"):
        return text
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def _run_moondream(best_frames: list[str]) -> GameDNA | None:
    """
    Run best available Ollama vision model (llava preferred, moondream fallback).
    LLaVA is much better at structured JSON output than Moondream.
    Install: ollama pull llava
    """
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests not installed")

    ollama_url = "http://localhost:11434/api/generate"

    # llava is the most reliable for structured JSON output
    # bakllava removed — returns single character responses for this prompt style
    # moondream kept as last resort fallback
    available_model = None
    for model in ["llava", "moondream"]:
        try:
            r = requests.post("http://localhost:11434/api/show",
                              json={"name": model}, timeout=5)
            if r.status_code == 200:
                available_model = model
                print(f"[VLM] Using model: {model}")
                break
        except Exception:
            continue

    if not available_model:
        print("[VLM] No vision model found. Run: ollama pull llava")
        return None

    # 3 frames — better genre detection, especially for games with distinct scenes
    # (Pokemon has overworld + battle, KH has exploration + combat)
    frames_to_use = best_frames[:3]
    images_b64 = []
    for frame_path in frames_to_use:
        with open(frame_path, "rb") as f:
            images_b64.append(base64.b64encode(f.read()).decode())
    print(f"[VLM] Sending {len(images_b64)} frames to {available_model}")

    PROMPT = """You are a video game expert. Look at these game screenshots carefully.

Reply with ONLY this JSON object, filling in each field. No other text:

{
  "title_guess": "your best guess at the game name",
  "genre": "wave_shooter",
  "setting": "describe the world in one sentence",
  "color_palette": ["#000000", "#000000", "#000000", "#000000"],
  "player_description": "describe the main character",
  "enemy_description": "describe the main enemy type",
  "boss_description": "describe the boss or villain",
  "environment_description": "describe the level environment",
  "music_vibe": "intense_action",
  "music_tempo": "fast",
  "confidence": 0.8
}

IMPORTANT for genre — use these rules in order, pick the FIRST one that matches:
1. If battle MENU with FIGHT/ATTACK/ITEM commands visible -> turn_based_rpg
2. If CARS or VEHICLES driving in a CITY with a MINIMAP -> open_world_sandbox
3. If SIDE-VIEW camera with player JUMPING between PLATFORMS -> side_scroll_platformer
4. If player uses a SWORD, KEYBLADE, or MAGIC SPELLS + companion characters follow + there is world exploration -> top_down_action_rpg
5. If HUD shows a WAVE NUMBER or ROUND COUNTER + player fires a GUN with ammo -> wave_shooter
6. If none match clearly, pick top_down_action_rpg as default for action games.
NOTE: Kingdom Hearts = top_down_action_rpg. Zelda = top_down_action_rpg. Diablo = top_down_action_rpg.

IMPORTANT for music_vibe — pick exactly one:
intense_action, dark_horror, epic_adventure, urban_gritty, mysterious

IMPORTANT for music_tempo — pick exactly one:
slow, medium, fast, frantic

Replace all placeholder values with real observations from the screenshots."""

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                ollama_url,
                json={
                    "model":  available_model,
                    "prompt": PROMPT,
                    "images": images_b64,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 800}
                },
                timeout=480  # 8 minutes — enough for 3 frames on any GPU
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            raw = _extract_json_from_text(raw)
            print(f"[VLM] Raw output (attempt {attempt+1}): {raw[:150]}...")

            dna = validate_vlm_output(raw, attempt)
            if dna:
                return dna

        except requests.exceptions.ConnectionError:
            print("[VLM] Ollama not running. Start with: ollama serve")
            return None
        except Exception as e:
            print(f"[VLM] Attempt {attempt + 1} error: {e}")

    print("[VLM] All retries exhausted")
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