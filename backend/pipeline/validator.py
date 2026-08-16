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
        "side_scroll_platformer",
        "turn_based_rpg"
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
    "turn_based_rpg": {
        "title_guess": "EPIC QUEST",
        "setting": "fantasy world with magical creatures and ancient dungeons",
        "color_palette": ["#1a0a2e", "#2d1b69", "#7b2d8b", "#f0c040"],
        "player_description": "young hero in fantasy armor holding a sword, front view",
        "enemy_description": "dark creature or monster with glowing eyes, front view",
        "boss_description": "massive dragon or demon lord with wings and fire",
        "environment_description": "dark battle arena with stone floor and magical effects",
        "music_vibe": "epic_adventure",
        "music_tempo": "medium",
    },
    "side_scroll_platformer": {
        "title_guess": "HERO QUEST",
        "setting": "colorful platformer world with grassy hills",
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


def _fuzzy_fix(data: dict) -> dict:
    """
    Fix common Moondream enum mismatches before Pydantic validation.
    Small models often get the right idea but wrong exact string.
    """
    # Genre fuzzy map
    genre_map = {
        "top down": "top_down_action_rpg", "top-down": "top_down_action_rpg",
        "action rpg": "top_down_action_rpg", "rpg": "top_down_action_rpg",
        "zelda": "top_down_action_rpg", "kingdom hearts": "top_down_action_rpg", "kh": "top_down_action_rpg", "hack and slash": "top_down_action_rpg", "action adventure": "top_down_action_rpg",
        "wave": "wave_shooter", "shooter": "wave_shooter", "horde": "wave_shooter",
        "zombie": "wave_shooter", "survival": "wave_shooter",
        "open world": "open_world_sandbox", "sandbox": "open_world_sandbox",
        "gta": "open_world_sandbox", "driving": "open_world_sandbox",
        "platform": "side_scroll_platformer", "platformer": "side_scroll_platformer",
        "side scroll": "side_scroll_platformer", "mario": "side_scroll_platformer",
        "turn based": "turn_based_rpg", "turn-based": "turn_based_rpg",
        "pokemon": "turn_based_rpg", "final fantasy": "turn_based_rpg",
        "jrpg": "turn_based_rpg", "battle menu": "turn_based_rpg",
        "undertale": "turn_based_rpg", "party battle": "turn_based_rpg",
    }
    # Vibe fuzzy map
    vibe_map = {
        "intensive action": "intense_action", "intense": "intense_action",
        "action": "intense_action", "fast": "intense_action",
        "dark": "dark_horror", "horror": "dark_horror", "scary": "dark_horror",
        "epic": "epic_adventure", "adventure": "epic_adventure", "heroic": "epic_adventure",
        "urban": "urban_gritty", "gritty": "urban_gritty", "city": "urban_gritty",
        "mystery": "mysterious", "mysterious": "mysterious", "eerie": "mysterious",
    }
    # Tempo fuzzy map
    tempo_map = {
        "very fast": "frantic", "very slow": "slow", "moderate": "medium",
        "quick": "fast", "rapid": "fast", "frantic": "frantic",
    }

    def fuzzy_match(value: str, mapping: dict, valid: list) -> str:
        if not isinstance(value, str):
            return valid[0]
        v = value.lower().strip()
        if v in valid:
            return v
        for key, mapped in mapping.items():
            if key in v:
                return mapped
        return valid[0]

    VALID_GENRES = ["wave_shooter","top_down_action_rpg","open_world_sandbox","side_scroll_platformer","turn_based_rpg"]
    VALID_VIBES  = ["intense_action","dark_horror","epic_adventure","urban_gritty","mysterious"]
    VALID_TEMPOS = ["slow","medium","fast","frantic"]

    if "genre" in data:
        data["genre"] = fuzzy_match(data["genre"], genre_map, VALID_GENRES)
    if "music_vibe" in data:
        data["music_vibe"] = fuzzy_match(data["music_vibe"], vibe_map, VALID_VIBES)
    if "music_tempo" in data:
        data["music_tempo"] = fuzzy_match(data["music_tempo"], tempo_map, VALID_TEMPOS)

    # Fix confidence if it comes back as a list (Moondream bug on attempt 2+)
    if isinstance(data.get("confidence"), list):
        data["confidence"] = 0.5

    return data


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
        data = _fuzzy_fix(data)   # Fix common model enum mismatches
        dna = GameDNA.model_validate(data)
        print(f"[Validator] [OK] Valid GameDNA (attempt {attempt + 1}): "
              f"genre={dna.genre}, confidence={dna.confidence:.2f}")
        return dna
    except Exception as e:
        print(f"[Validator] [FAIL] Attempt {attempt + 1} failed: {e}")
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