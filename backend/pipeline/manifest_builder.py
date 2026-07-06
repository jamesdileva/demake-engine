"""
Sprint 4 — Manifest Assembly

Takes all generated assets (sprites, audio, game config) and assembles
the final manifest.json that the Phaser frontend reads to boot the game.

This is the glue between the ML pipeline and the game engine.
The manifest contract is defined in ARCHITECTURE.md Section 11.
"""
import os
import json
from datetime import datetime

from database.db import SessionLocal
from database.models import Demake, GameConfig, Asset
from pipeline.validator import GameDNA


def build_manifest(demake_id: str, dna: GameDNA,
                   template_id: str, output_dir: str) -> dict:
    """
    Assemble the complete game manifest from DB records and generated files.

    Args:
        demake_id:   UUID of the demake run
        dna:         Validated GameDNA from VLM analysis
        template_id: Matched genre template (e.g. "wave_shooter")
        output_dir:  /outputs/{demake_id}/

    Returns:
        The complete manifest dict (also written to manifest.json)
    """
    db = SessionLocal()
    try:
        demake     = db.query(Demake).filter_by(id=demake_id).first()
        game_cfg   = db.query(GameConfig).filter_by(demake_id=demake_id).first()
        assets     = db.query(Asset).filter_by(demake_id=demake_id).all()
    finally:
        db.close()

    # ── Build asset URL map ───────────────────────────────────────────────────
    # Maps slot_name → API URL the frontend will fetch
    # e.g. "sprite_player" → "/api/v1/demake/{id}/asset/sprite_player.png"

    sprite_urls = {}
    sprite_meta = {}
    audio_urls  = {}

    for asset in assets:
        if not asset.file_path or not os.path.exists(asset.file_path):
            continue

        filename = os.path.basename(asset.file_path)
        asset_url = f"/api/v1/demake/{demake_id}/asset/{filename}"

        if asset.asset_type.startswith("audio_"):
            track_name = asset.slot_name or asset.asset_type.replace("audio_", "")
            audio_urls[track_name] = asset_url

        elif asset.asset_type.startswith(("sprite_", "tile_")):
            slot = asset.asset_type
            sprite_urls[slot] = asset_url

            # Build animation metadata from slot config
            sprite_meta[slot] = _build_sprite_meta(slot, asset)

    # ── Build palette dict ────────────────────────────────────────────────────
    palette = dna.color_palette
    palette_dict = {
        "primary":    palette[0] if len(palette) > 0 else "#1a0a00",
        "secondary":  palette[1] if len(palette) > 1 else "#3d2b1f",
        "accent":     palette[2] if len(palette) > 2 else "#8b1a00",
        "highlight":  palette[3] if len(palette) > 3 else "#ff4400",
        "background": _darken_hex(palette[0] if palette else "#1a0a00"),
        "hud_text":   palette[3] if len(palette) > 3 else "#ff4400",
        "hud_dim":    _darken_hex(palette[2] if len(palette) > 2 else "#8b1a00"),
    }

    # ── Build game config from template + DNA ─────────────────────────────────
    game_config = _build_game_config(template_id, dna)

    # ── Include tilemap if generated ──────────────────────────────────────────
    tilemap_path = os.path.normpath(os.path.join(output_dir, "tilemap.json"))
    tilemap_ref  = None
    if os.path.exists(tilemap_path):
        with open(tilemap_path) as f:
            tilemap_ref = json.load(f)
        print(f"[Manifest] Tilemap included: {tilemap_ref['width']}x{tilemap_ref['height']}")

    # ── Assemble final manifest ───────────────────────────────────────────────
    manifest = {
        "demake_id":         demake_id,
        "title":             (demake.title or dna.title_guess or "UNTITLED").upper(),
        "generated_at":      datetime.utcnow().isoformat() + "Z",
        "source_game_guess": dna.title_guess,
        "template":          template_id,
        "setting":           dna.setting,

        "palette": palette_dict,

        "game_config": game_config,

        "tilemap": tilemap_ref,

        "assets": {
            "sprites":     sprite_urls,
            "sprite_meta": sprite_meta,
            "audio":       audio_urls,
        },

        "debug": {
            "vlm_confidence":          dna.confidence,
            "music_vibe":              dna.music_vibe,
            "music_tempo":             dna.music_tempo,
            "player_description":      dna.player_description,
            "enemy_description":       dna.enemy_description,
            "generation_completed_at": datetime.utcnow().isoformat() + "Z",
        }
    }

    # ── Write to disk ─────────────────────────────────────────────────────────
    manifest_path = os.path.normpath(os.path.join(output_dir, "manifest.json"))
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Manifest] Written to {manifest_path}")
    print(f"[Manifest] Sprites: {len(sprite_urls)} | Audio: {len(audio_urls)}")

    return manifest


def _build_game_config(template_id: str, dna: GameDNA) -> dict:
    """
    Build the game_config block by merging the template defaults
    with any DNA-driven tweaks (e.g. speed from music tempo).
    """
    # Base configs per template — matches the template JSON files
    # from the architecture doc Section 4
    TEMPLATE_CONFIGS = {
        "wave_shooter": {
            "physics":  {"gravity": 0, "player_speed": 130, "bullet_speed": 380},
            "waves":    {"start_count": 6, "multiplier": 1.4,
                         "boss_every": 5, "spawn_delay_ms": 700},
            "player":   {"max_hp": 100, "max_ammo": 30,
                         "reload_time_ms": 1800, "fire_rate_ms": 150},
            "enemy":    {"base_hp": 30, "base_speed": 55, "damage": 10,
                         "hp_scale": 1.2, "speed_scale": 1.05},
            "boss":     {"hp_mult": 8, "speed_mult": 0.6,
                         "damage_mult": 3, "size_mult": 2.0},
            "hud":      ["hp_bar", "ammo_counter", "wave_number", "score"],
        },
        "top_down_action_rpg": {
            "physics":  {"gravity": 0, "player_speed": 110, "bullet_speed": 0},
            "combat":   {"max_hp": 120, "max_mp": 60, "attack_damage": 25,
                         "attack_range": 40, "dodge_cooldown_ms": 800},
            "enemy":    {"base_hp": 50, "base_speed": 45, "damage": 15,
                         "aggro_range": 120},
            "boss":     {"hp_mult": 10, "speed_mult": 0.7, "damage_mult": 2.5},
            "hud":      ["hp_bar", "mp_bar", "score", "boss_hp"],
        },
        "open_world_sandbox": {
            "physics":  {"gravity": 0, "player_speed": 100,
                         "car_speed": 220, "friction": 0.85},
            "world":    {"wanted_max": 5, "npc_count": 12,
                         "mission_count": 3},
            "player":   {"max_hp": 100, "cash": 0},
            "hud":      ["hp_bar", "wanted_stars", "cash", "minimap"],
        },
        "side_scroll_platformer": {
            "physics":  {"gravity": 800, "player_speed": 160,
                         "jump_velocity": -420, "bullet_speed": 300},
            "player":   {"lives": 3, "max_hp": 3, "invincible_ms": 1500},
            "enemy":    {"base_hp": 1, "base_speed": 40, "damage": 1},
            "hud":      ["lives", "score", "coins", "timer"],
        },
        "turn_based_rpg": {
            "physics":  {"gravity": 0, "player_speed": 0},
            "battle":   {"party_size": 3, "enemy_count": 3,
                         "player_hp": 80, "player_mp": 40,
                         "base_attack": 20, "base_magic": 30, "base_heal": 25},
            "enemy":    {"base_hp": 60, "base_attack": 15, "base_speed": 50},
            "boss":     {"hp_mult": 5, "attack_mult": 2},
            "hud":      ["hp_bar", "mp_bar", "turn_order", "action_menu"],
        },
    }

    cfg = TEMPLATE_CONFIGS.get(template_id,
                               TEMPLATE_CONFIGS["wave_shooter"]).copy()

    # DNA-driven tweaks — faster music = faster gameplay
    speed_mult = {
        "slow": 0.75, "medium": 1.0, "fast": 1.2, "frantic": 1.4
    }.get(dna.music_tempo, 1.0)

    if "physics" in cfg and "player_speed" in cfg["physics"]:
        cfg["physics"]["player_speed"] = int(
            cfg["physics"]["player_speed"] * speed_mult
        )
    if "enemy" in cfg and "base_speed" in cfg["enemy"]:
        cfg["enemy"]["base_speed"] = int(
            cfg["enemy"]["base_speed"] * speed_mult
        )

    return cfg


def _build_sprite_meta(slot_name: str, asset: Asset) -> dict:
    """
    Build animation metadata for a sprite slot.
    Tells Phaser how to split the spritesheet and which frames = which state.
    """
    # Default animation configs per slot
    ANIM_CONFIGS = {
        "sprite_player": {
            "frame_width": 16, "frame_count": 8,
            "animations": {
                "idle":  {"frames": [0, 1],       "fps": 4,  "loop": True},
                "walk":  {"frames": [2, 3, 4, 5], "fps": 8,  "loop": True},
                "shoot": {"frames": [6, 7],        "fps": 12, "loop": False},
            }
        },
        "sprite_enemy": {
            "frame_width": 16, "frame_count": 4,
            "animations": {
                "walk":   {"frames": [0, 1, 2, 3], "fps": 6,  "loop": True},
                "attack": {"frames": [2, 3],        "fps": 10, "loop": False},
            }
        },
        "sprite_boss": {
            "frame_width": 32, "frame_count": 6,
            "animations": {
                "idle":   {"frames": [0, 1],       "fps": 3,  "loop": True},
                "walk":   {"frames": [2, 3],       "fps": 5,  "loop": True},
                "attack": {"frames": [4, 5],       "fps": 10, "loop": False},
            }
        },
        "sprite_projectile": {
            "frame_width": 8, "frame_count": 2,
            "animations": {
                "fly": {"frames": [0, 1], "fps": 8, "loop": True},
            }
        },
        "tile_floor": {
            "frame_width": 16, "frame_count": 1, "animations": {}
        },
        "tile_wall": {
            "frame_width": 16, "frame_count": 1, "animations": {}
        },
    }

    return ANIM_CONFIGS.get(slot_name, {
        "frame_width": asset.frame_width or 16,
        "frame_count": asset.frame_count or 1,
        "animations": {}
    })


def _darken_hex(hex_color: str, factor: float = 0.5) -> str:
    """Darken a hex color by a factor (0=black, 1=original)."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return "#000000"