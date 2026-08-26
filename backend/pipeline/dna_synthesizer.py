"""
Sprint 8F — Genre-Only DNA Synthesizer (Mode -1)

Builds a fully valid GameDNA from genre template defaults + seeded random
variation, so repeated runs of the same genre produce different-looking
worlds (palette hue-shift, title, setting, music vibe).

Replaces the ingestion + VLM stages for genre-only jobs:
zero cloud cost, zero video processing, instant start.

Architecture doc reference (Sprint 8F):
  POST /api/v1/demake/generate { genre } -> synthesized game_dna.json
"""
import colorsys
import random

from pipeline.validator import GameDNA, GENRE_DEFAULTS


# ── Per-genre flavor tables ────────────────────────────────────────────────────

TITLE_PARTS = {
    "wave_shooter": (
        ["DEAD", "CRIMSON", "IRON", "BLACK", "LAST", "BLOOD"],
        ["BUNKER", "OUTPOST", "ZONE", "SIEGE", "PROTOCOL", "HORDE"],
    ),
    "top_down_action_rpg": (
        ["SHADOW", "CRYSTAL", "DRAGON", "RUIN", "MOONLIGHT", "EMBER"],
        ["QUEST", "BLADE", "REALM", "LEGEND", "DEPTHS", "KEEP"],
    ),
    "open_world_sandbox": (
        ["GRAND", "NEON", "MIDNIGHT", "ASPHALT", "SINNER"],
        ["CITY", "HEIST", "STREETS", "EMPIRE", "RUN"],
    ),
    "turn_based_rpg": (
        ["EPIC", "POCKET", "ARCANE", "FINAL", "ETERNAL", "MYSTIC"],
        ["QUEST", "MONSTERS", "SAGA", "ODYSSEY", "ARENA", "TALES"],
    ),
    "side_scroll_platformer": (
        ["SUPER", "PIXEL", "TURBO", "JUMPING", "CAVE", "CLOUD"],
        ["HERO", "LEAP", "WORLD", "RUNNER", "QUEST", "HOP"],
    ),
}

SETTINGS = {
    "wave_shooter": [
        "abandoned military bunker overrun with undead",
        "crimson-lit industrial compound crawling with mutants",
        "frozen outpost besieged by night creatures",
        "collapsing research facility full of specimens",
    ],
    "top_down_action_rpg": [
        "cursed castle with torch-lit stone corridors",
        "enchanted forest hiding a ruined shrine",
        "floating islands above a sea of clouds",
        "volcanic keep guarded by armored knights",
    ],
    "open_world_sandbox": [
        "rain-soaked neon city streets at midnight",
        "sun-baked downtown with gridlocked traffic",
        "harbor district run by rival gangs",
        "sprawling metropolis with rooftop chases",
    ],
    "turn_based_rpg": [
        "whimsical meadow routes full of wild creatures",
        "ancient continent of elemental gyms and ruins",
        "moonlit kingdom of monsters and magic",
        "sky islands linked by rainbow bridges",
    ],
    "side_scroll_platformer": [
        "sunny hills with floating platforms and pipes",
        "underground caverns glittering with crystals",
        "cloud kingdom above the treetops",
        "candy world of bouncy jelly platforms",
    ],
}

VIBES = {
    "wave_shooter":        ["dark_horror", "intense_action"],
    "top_down_action_rpg": ["epic_adventure", "mysterious"],
    "open_world_sandbox":  ["urban_gritty", "intense_action"],
    "turn_based_rpg":      ["epic_adventure", "mysterious"],
    "side_scroll_platformer": ["epic_adventure", "intense_action"],
}

TEMPOS = {
    "wave_shooter":           ["fast", "frantic"],
    "top_down_action_rpg":    ["medium", "fast"],
    "open_world_sandbox":     ["medium", "fast"],
    "turn_based_rpg":         ["medium", "slow"],
    "side_scroll_platformer": ["fast", "medium"],
}


def _shift_palette(palette: list[str], rng: random.Random) -> list[str]:
    """
    Hue-rotate the whole palette by one shared random offset (keeps colors
    harmonious) with a small per-color jitter. Returns valid #RRGGBB strings.
    """
    hue_shift = rng.uniform(0.0, 1.0)
    shifted = []
    for hexc in palette:
        try:
            r = int(hexc[1:3], 16) / 255.0
            g = int(hexc[3:5], 16) / 255.0
            b = int(hexc[5:7], 16) / 255.0
        except (ValueError, IndexError):
            shifted.append(hexc)
            continue
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        h = (h + hue_shift + rng.uniform(-0.03, 0.03)) % 1.0
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        shifted.append("#%02x%02x%02x" % (
            int(r * 255), int(g * 255), int(b * 255)))
    return shifted


def synthesize_dna(genre: str, seed=None) -> GameDNA:
    """
    Build a fully valid GameDNA for the given genre.

    Same seed + genre => identical DNA (deterministic, great for testing).
    seed=None => fresh random world each call.
    """
    rng = random.Random(seed)

    base = GENRE_DEFAULTS.get(genre, GENRE_DEFAULTS["wave_shooter"]).copy()

    prefixes, suffixes = TITLE_PARTS.get(
        genre, TITLE_PARTS["wave_shooter"])
    base["title_guess"] = f"{rng.choice(prefixes)} {rng.choice(suffixes)}"
    base["setting"] = rng.choice(
        SETTINGS.get(genre, SETTINGS["wave_shooter"]))
    base["color_palette"] = _shift_palette(
        base["color_palette"], rng)
    base["music_vibe"] = rng.choice(VIBES.get(genre, VIBES["wave_shooter"]))
    base["music_tempo"] = rng.choice(TEMPOS.get(genre, TEMPOS["wave_shooter"]))
    base["confidence"] = 1.0  # synthesized, not guessed

    print(f"[DNASynth] genre={genre} seed={seed} "
          f"title='{base['title_guess']}' vibe={base['music_vibe']}")
    return GameDNA(genre=genre, **base)
