"""
Sprint 2 — Pydantic Validation + Retry Logic

The VLM sometimes returns malformed JSON or uses wrong enum values.
This module validates the output and re-prompts up to 3 times before
falling back to safe defaults. The pipeline never crashes due to bad VLM output.

Architecture doc reference:
  VLM output → schema validation (Pydantic) →
  if invalid, re-prompt with error context →
  max 3 retries → fallback to defaults
"""
from typing import Literal
from pydantic import BaseModel, Field, field_validator


# ── Game DNA schema ────────────────────────────────────────────────────────────
# This is the contract between the VLM and the rest of the pipeline.
# Every field maps directly to a key in the VLM prompt.

class GameDNA(BaseModel):
    """
    The extracted 'DNA' of a game, parsed from VLM analysis of trailer keyframes.
    All fields have safe defaults so a partial VLM response still produces a result.
    """
    title_guess: str = "UNKNOWN GAME"

    genre: Literal[
        "wave_shooter",
        "top_down_action_rpg",
        "open_world_sandbox",
        "side_scroll_platformer"
    ] = "wave_shooter"

    setting: str = "mysterious environment"

    color_palette: list[str] = Field(
        default=["#1a0a00", "#3d2b1f", "#8b1a00", "#ff4400"]
    )

    player_description: str = "armored hero character, side view"
    enemy_description:  str = "menacing enemy creature"
    boss_description:   str = "massive powerful boss enemy"
    environment_description: str = "dark atmospheric environment with walls and floor"

    music_vibe: Literal[
        "intense_action",
        "dark_horror",
        "epic_adventure",
        "urban_gritty",
        "mysterious"
    ] = "intense_action"

    music_tempo: Literal["slow", "medium", "fast", "frantic"] = "fast"

    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("color_palette")
    @classmethod
    def validate_palette(cls, v):
        """Ensure we have 4-6 valid hex codes. Fill with defaults if short."""
        defaults = ["#1a0a00", "#3d2b1f", "#8b1a00", "#ff4400", "#ffcc00", "#ffffff"]
        cleaned = []
        for color in v:
            c = color.strip()
            if not c.startswith("#"):
                c = "#" + c
            if len(c) == 7:  # Valid #RRGGBB
                cleaned.append(c)
        # Pad to minimum 4 colors
        while len(cleaned) < 4:
            cleaned.append(defaults[len(cleaned)])
        return cleaned[:6]  # Max 6


# ── Safe defaults per genre ────────────────────────────────────────────────────
GENRE_DEFAULTS: dict[str, dict] = {
    "wave_shooter": {
        "title_guess": "ZOMBIE SURVIVAL",
        "setting": "abandoned military base overrun with undead",
        "color_palette": ["#1a0a00", "#3d2b1f", "#8b1a00", "#ff4400"],
        "player_description": "armored soldier in military fatigues, holding rifle, side view",
        "enemy_description": "shambling zombie in torn clothes, outstretched arms",
        "boss_description": "massive zombie brute with oversized fists and glowing eyes",
        "environment_description": "dark concrete bunker walls with metal floor grates",
        "music_vibe": "dark_horror",
        "music_tempo": "frantic",
    },
    "top_down_action_rpg": {
        "title_guess": "SHADOW QUEST",
        "setting": "fantasy world with castles and dark forests",
        "color_palette": ["#0a0a1a", "#1a1a4a", "#4a3a8a", "#ff6600"],
        "player_description": "young hero in fantasy armor with sword, top-down view",
        "enemy_description": "dark shadow creature with glowing eyes",
        "boss_description": "enormous dragon or demon lord with wings",
        "environment_description": "stone dungeon floor with torch-lit corridors",
        "music_vibe": "epic_adventure",
        "music_tempo": "medium",
    },
    "open_world_sandbox": {
        "title_guess": "CITY CHAOS",
        "setting": "gritty urban city streets",
        "color_palette": ["#0a0a0a", "#2a2a2a", "#4a4a4a", "#ffcc00"],
        "player_description": "street tough in jacket and jeans, top-down view",
        "enemy_description": "police officer or rival gang member",
        "boss_description": "crime boss in suit with bodyguards",
        "environment_description": "city street with cars, sidewalks and buildings",
        "music_vibe": "urban_gritty",
        "music_tempo": "medium",
    },
    "side_scroll_platformer": {
        "title_guess": "HERO QUEST",
        "setting": "colorful platformer world",
        "color_palette": ["#0a2a6a", "#1a4a9a", "#4a8aff", "#ffff00"],
        "player_description": "small hero character in cap and boots, side view",
        "enemy_description": "round bouncing creature with eyes",
        "boss_description": "large mechanical or monster boss enemy",
        "environment_description": "grassy platforms with clouds and hills in background",
        "music_vibe": "epic_adventure",
        "music_tempo": "fast",
    },
}


def get_safe_defaults(genre: str = "wave_shooter") -> GameDNA:
    """Return a fully valid GameDNA using hardcoded defaults for the given genre."""
    defaults = GENRE_DEFAULTS.get(genre, GENRE_DEFAULTS["wave_shooter"])
    return GameDNA(genre=genre, confidence=0.1, **defaults)


def validate_vlm_output(raw_json: str, attempt: int = 0) -> GameDNA | None:
    """
    Try to parse raw VLM JSON output into a validated GameDNA object.

    Returns GameDNA on success, None on failure (caller should retry).
    """
    import json

    # Clean common VLM output issues
    cleaned = raw_json.strip()
    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            l for l in lines
            if not l.strip().startswith("```")
        )

    try:
        data = json.loads(cleaned)
        dna = GameDNA.model_validate(data)
        print(f"[Validator] ✓ Valid GameDNA (attempt {attempt + 1}): "
              f"genre={dna.genre}, confidence={dna.confidence:.2f}")
        return dna
    except Exception as e:
        print(f"[Validator] ✗ Attempt {attempt + 1} failed: {e}")
        return None


def build_retry_prompt(original_output: str, error: str) -> str:
    """
    Build a correction prompt to send back to the VLM when validation fails.
    Tells the VLM exactly what was wrong so it can fix it.
    """
    return f"""Your previous response was not valid JSON or did not match the required schema.

Error: {error}

Your previous response was:
{original_output[:500]}

Please try again. You MUST respond with ONLY valid JSON.
No explanation, no markdown code blocks, no preamble. Raw JSON only.

The "genre" field MUST be exactly one of:
  "wave_shooter" | "top_down_action_rpg" | "open_world_sandbox" | "side_scroll_platformer"

The "music_vibe" field MUST be exactly one of:
  "intense_action" | "dark_horror" | "epic_adventure" | "urban_gritty" | "mysterious"

The "music_tempo" field MUST be exactly one of:
  "slow" | "medium" | "fast" | "frantic"
"""