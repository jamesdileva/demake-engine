# 🎮 DEMAKE ENGINE — Grand Architecture Document
### *Turn any modern game trailer into a playable 8-bit browser game*
> Version 1.1 | Status: Sprint 8A In Progress | Target: GitHub Open Source

---

## Table of Contents
1. [Vision & Scope](#1-vision--scope)
2. [System Overview](#2-system-overview)
3. [Hardware Strategy](#3-hardware-strategy)
4. [Genre Template Library](#4-genre-template-library-sprint-0)
5. [Backend & API](#5-backend--api)
6. [Database Schema](#6-database-schema)
7. [ML Pipeline](#7-ml-pipeline-the-brain)
8. [Asset Generation Pipeline](#8-asset-generation-pipeline)
9. [Audio Pipeline](#9-audio-pipeline)
10. [Frontend Game Engine](#10-frontend-game-engine)
11. [The JSON Manifest Contract](#11-the-json-manifest-contract)
12. [Sprint Plan](#12-sprint-plan)
13. [Tech Stack Summary](#13-tech-stack-summary)
14. [Future Roadmap](#14-future-roadmap)

---

## 1. Vision & Scope

### What This Is
You feed the system a **genre pick** (or optionally a modern game trailer MP4). A multimodal AI pipeline extracts the game's DNA — genre, color palette, character shapes, mechanics, and vibe — and automatically generates a **playable 8-bit browser demake** complete with procedurally generated chiptune music.

### Two Input Modes (v1.1)
| Mode | Input | Pipeline Path | Status |
|---|---|---|---|
| **Mode -1: Genre-Only** *(default)* | Pick a genre from a dropdown | Template defaults + seeded random DNA → sprites → audio → WFC tilemap → playable world slice. No upload, no VLM call, no video. | Sprint 8F |
| **Mode 2: Trailer Upload** *(secondary)* | Upload an MP4 trailer | Full VLM pipeline as originally designed. Treated as a "vibe seed" — it was never going to recreate the exact source game; it produces a demake *inspired by* the trailer's DNA. | Working since Sprint 4 |

Genre-Only is the headline flow: it is instant, free, and showcases the procedural core of the engine. Trailer mode stays available for when you want AI-derived art direction from real footage.

### Target Games (Inspirations)
These three genres define the scope of v1. They are intentionally very different, which forces the system to be genre-flexible by design.

| Inspiration | Genre Template | Core Mechanic |
|---|---|---|
| Kingdom Hearts | Action RPG (top-down) | Melee combat, MP bar, enemy HP |
| CoD Zombies | Wave-based Shooter | Ammo, wave counter, barricades |
| GTA | Open World / Sandbox | Free movement, wanted level, missions |

### What This Is NOT
- Not a full game engine replacement
- Not training new ML models from scratch
- Not a commercial product (open source, for fun & learning)
- Not guaranteed to be pixel-perfect to the source material

---

## 2. System Overview

```
[ User picks a genre ]  ── or ──  [ User uploads trailer MP4 ]
          │ Mode -1: synthesized DNA           │
          │ (template defaults + seed)         │
          ▼                                    ▼
┌─────────────────────┐
│   FastAPI Backend   │  ← Handles upload, queuing, status polling
│   (Orchestrator)    │
└────────┬────────────┘
          │ Enqueues job
          ▼
┌─────────────────────────────────────────────────────┐
│                  ML PIPELINE                         │
│                                                      │
│  1. ffmpeg         → Extract keyframes + audio      │
│     (trailer mode only — skipped in genre-only)      │
│  2. VLM (Cloud)    → Analyze frames → Game DNA JSON  │
│     (trailer mode only — skipped in genre-only)      │
│  3. Genre Matcher  → Select template                │
│  4. Validator      → Pydantic schema check + retry  │
│  5. Sprite Gen     → Local SD → pixel art sheets    │
│  6. Audio Gen      → MIDI algorithmic generator     │
│  7. Logic Gen      → Game config JSON               │
│  8. WFC Tilemap    → Procedural level layout        │
└────────┬────────────────────────────────────────────┘
         │ Writes assets + manifest to disk + DB
         ▼
┌─────────────────────┐
│     SQLite DB       │  ← Tracks state, stores paths
└────────┬────────────┘
         │ Browser polls status
         ▼
┌─────────────────────┐
│   Browser Client    │  ← Fetches manifest, boots game engine
│   Phaser.js Engine  │  ← Renders the 8-bit demake
└─────────────────────┘
```

### Data Flow Summary
```
MP4 → keyframes (PNG) + audio (WAV)
    → VLM analysis → game_dna.json
    → genre template selected
    → sprites generated (PNG spritesheets)
    → MIDI generated (.mid)
    → game_manifest.json assembled
    → Browser loads manifest → playable game
```

---

## 3. Hardware Strategy

### The Constraint
- **GPU VRAM:** ~4GB (tight but workable)
- **Virtual Memory:** 27GB (useful for CPU offloading)
- **Goal:** Minimize cloud API cost to near-zero for most steps

### Strategy: Hybrid Local/Cloud

| Pipeline Step | Approach | Why |
|---|---|---|
| ffmpeg extraction | Local | Free, CPU-based, no VRAM needed |
| VLM frame analysis | **Cloud (GPT-4o mini)** | 4GB VRAM can't run a quality VLM; GPT-4o mini API calls are tiny & cheap (~$0.01/trailer) |
| Sprite generation | Local (SD 1.5 LCM, 8-bit) | Fits in 4GB VRAM at INT8 precision |
| Audio generation | Local (algorithmic MIDI) | Zero VRAM — pure Python math |
| Logic/config generation | Local (Llama-3-8B at 4-bit) | ~3.5GB VRAM; use llama.cpp for CPU offload fallback |

### VRAM Management Rules
- Never load sprite model and LLM simultaneously — pipeline is sequential, unload between steps
- Use `bfloat16` for SD 1.5, `Q4_K_M` quantization for Llama
- If VRAM is exceeded, fall back to CPU via llama.cpp (slower but won't crash)
- Keep a `config.yaml` flag: `force_cpu: true` for users with no GPU

---

## 4. Genre Template Library (Sprint 0)

The most important architectural decision: **the ML pipeline selects and parameterizes a template — it does NOT invent game logic.** This makes output reliable and scope manageable.

### The 4 Core Templates

#### Template A: `top_down_action_rpg`
*Kingdom Hearts / Zelda-style*
- **Camera:** Top-down, player centered
- **Movement:** 8-directional WASD
- **Mechanics:** HP bar, MP bar, melee attack (space), dodge roll (shift)
- **Win condition:** Defeat boss enemy
- **Enemies:** Patrol AI, aggro on proximity

#### Template B: `wave_shooter`
*CoD Zombies / Horde mode*
- **Camera:** Top-down or side-view
- **Movement:** WASD, aim with mouse
- **Mechanics:** Ammo counter, wave number HUD, reload (R), barriers (health-gated)
- **Win condition:** Survive N waves
- **Enemies:** Spawn from edges, pathfind toward player

#### Template C: `open_world_sandbox`
*GTA / top-down*
- **Camera:** Top-down, scrolling world
- **Movement:** Car physics OR foot movement, switch with E
- **Mechanics:** Wanted level stars (1-5), mission marker, cash counter
- **Win condition:** Complete mission objective
- **NPCs:** Random walk AI, flee on violence

#### Template D: `side_scroll_platformer`
*Generic fallback*
- **Camera:** Side-scrolling
- **Movement:** Left/right + jump
- **Mechanics:** Lives, coin counter, checkpoints
- **Win condition:** Reach end flag
- **Enemies:** Static patrol, stomp to kill

### Template Schema (stored as JSON files in `/templates/`)
```json
{
  "template_id": "wave_shooter",
  "display_name": "Wave Survival Shooter",
  "physics": {
    "gravity": 0,
    "player_speed": 120,
    "bullet_speed": 300,
    "friction": 0.85
  },
  "hud_elements": ["hp_bar", "ammo_counter", "wave_number", "score"],
  "spawn_config": {
    "enemy_spawn_edges": true,
    "wave_size_multiplier": 1.3,
    "boss_every_n_waves": 5
  },
  "controls": {
    "move": "WASD",
    "shoot": "mouse_click",
    "reload": "R",
    "interact": "E"
  },
  "camera": "top_down_follow",
  "sprite_slots": {
    "player": { "frames": 8, "states": ["idle","walk","shoot","die"] },
    "enemy_basic": { "frames": 4, "states": ["walk","attack","die"] },
    "enemy_boss": { "frames": 6, "states": ["idle","walk","attack","die"] },
    "tile_floor": { "frames": 1, "states": ["default"] },
    "tile_wall": { "frames": 1, "states": ["default"] },
    "projectile": { "frames": 2, "states": ["fly"] }
  }
}
```

---

## 5. Backend & API

### Framework: FastAPI + Python 3.11+

### Project Structure
```
demake-engine/
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.yaml              # Runtime config (VRAM limits, API keys, etc.)
│   ├── database/
│   │   ├── db.py                # SQLite connection + SQLAlchemy setup
│   │   └── models.py            # ORM models
│   ├── api/
│   │   └── routes/
│   │       ├── demake.py        # Upload, status, manifest endpoints
│   │       └── health.py        # Health check
│   ├── pipeline/
│   │   ├── orchestrator.py      # Async job runner
│   │   ├── ingestion.py         # ffmpeg extraction
│   │   ├── vlm_analysis.py      # Cloud VLM → game_dna.json
│   │   ├── genre_matcher.py     # DNA → template selection
│   │   ├── validator.py         # Pydantic schemas + retry logic
│   │   ├── sprite_gen.py        # SD 1.5 → pixel art spritesheets
│   │   ├── audio_gen.py         # Algorithmic MIDI generator
│   │   └── logic_gen.py         # LLM → game config JSON
│   └── templates/               # The 4 genre template JSON files
├── frontend/
│   ├── index.html               # Upload UI
│   ├── game.html                # Game engine host page
│   ├── engine/
│   │   ├── boot.js              # Loads manifest, boots Phaser scene
│   │   ├── scenes/
│   │   │   ├── PreloadScene.js  # Loads assets from manifest paths
│   │   │   ├── GameScene.js     # Main game loop (template-driven)
│   │   │   └── HUDScene.js      # Overlay UI (HP, ammo, score)
│   │   └── templates/           # JS implementations of the 4 templates
│   │       ├── TopDownRPG.js
│   │       ├── WaveShooter.js
│   │       ├── OpenWorld.js
│   │       └── Platformer.js
├── outputs/                     # Generated assets go here
│   └── {demake_id}/
│       ├── keyframes/
│       ├── sprites/
│       ├── audio/
│       └── manifest.json
└── requirements.txt
```

### Core API Endpoints

#### `POST /api/v1/demake/upload`
```
Payload:  multipart/form-data — video file (MP4, max 100MB)
Action:   Validate file, save to disk, create DB record, enqueue job
Returns:  { "demake_id": "uuid", "status": "queued" }
Errors:   413 if file too large, 415 if not MP4
```

#### `GET /api/v1/demake/{id}/status`
```
Action:   Polls SQLite for current pipeline stage
Returns:  {
            "status": "generating_sprites",
            "stage": 4,
            "total_stages": 7,
            "progress_pct": 57,
            "message": "Generating character spritesheets..."
          }
```

#### `GET /api/v1/demake/{id}/manifest`
```
Action:   Returns complete game manifest once status = "ready"
Returns:  Full game_manifest.json (see Section 11)
Errors:   409 if not yet complete, 404 if unknown ID
```

#### `GET /api/v1/demake/{id}/asset/{filename}`
```
Action:   Serves static generated files (sprites, audio) from outputs/
Returns:  Binary file (PNG, MID, etc.)
```

#### `WebSocket /ws/demake/{id}`
```
Action:   Real-time status stream during generation
Emits:    { "stage": "vlm_analysis", "pct": 22, "msg": "..." }
```

### Pipeline Stages & Status Labels
```
Stage 1: "queued"              → Job waiting in queue
Stage 2: "extracting_frames"   → ffmpeg running
Stage 3: "analyzing"           → VLM API call in progress
Stage 4: "matching_genre"      → Template selection + validation
Stage 5: "generating_sprites"  → Stable Diffusion running
Stage 6: "generating_audio"    → MIDI generation
Stage 7: "assembling"          → Manifest JSON being written
Stage 8: "ready"               → Done, manifest available
Stage X: "failed"              → Error stored in DB, reason returned
```

### Validation + Retry Loop (Critical)
```python
# validator.py — pseudocode
MAX_RETRIES = 3

async def validate_with_retry(vlm_output: str, schema: BaseModel) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            parsed = schema.model_validate_json(vlm_output)
            return parsed
        except ValidationError as e:
            if attempt == MAX_RETRIES - 1:
                return get_safe_defaults(schema)  # Never crash, use defaults
            vlm_output = await re_prompt_vlm(vlm_output, str(e))
    
    return get_safe_defaults(schema)
```

---

## 6. Database Schema

### SQLite with SQLAlchemy ORM | WAL Mode enabled

```sql
PRAGMA journal_mode=WAL;  -- Prevents lock contention with async workers
```

### Table: `demakes`
| Column | Type | Description |
|---|---|---|
| `id` | UUID (PK) | Unique run identifier |
| `title` | TEXT | Extracted or user-provided title |
| `status` | TEXT | Current pipeline stage label |
| `source_path` | TEXT | Path to uploaded MP4 |
| `error_message` | TEXT | NULL unless failed |
| `created_at` | DATETIME | Ingestion timestamp |
| `completed_at` | DATETIME | NULL until ready |

### Table: `game_configs`
| Column | Type | Description |
|---|---|---|
| `id` | UUID (PK) | Config identifier |
| `demake_id` | UUID (FK) | Links to demakes |
| `template_id` | TEXT | Which template was selected |
| `genre` | TEXT | Human-readable genre label |
| `color_palette` | TEXT | JSON array of 4-8 hex codes |
| `mechanics` | TEXT | JSON object (speed, gravity, health, etc.) |
| `hud_elements` | TEXT | JSON array of HUD items |
| `vlm_raw_output` | TEXT | Raw VLM response (for debugging) |

### Table: `assets`
| Column | Type | Description |
|---|---|---|
| `id` | UUID (PK) | Asset identifier |
| `demake_id` | UUID (FK) | Links to demakes |
| `asset_type` | TEXT | See types below |
| `slot_name` | TEXT | Template slot this fills (e.g. `sprite_player`) |
| `file_path` | TEXT | Local path to file |
| `frame_count` | INTEGER | For spritesheets |
| `frame_width` | INTEGER | Pixels per frame |
| `animation_states` | TEXT | JSON array of state names |
| `created_at` | DATETIME | Generation timestamp |

#### Asset Type Enum
```
sprite_player      — Player character spritesheet
sprite_enemy       — Enemy spritesheet
sprite_boss        — Boss enemy spritesheet
sprite_projectile  — Bullet / fireball
tile_floor         — Background floor tile
tile_wall          — Wall / obstacle tile
tile_decoration    — Non-collidable detail
audio_bgm          — Background music MIDI
audio_sfx_hit      — Hit sound effect
audio_sfx_shoot    — Shoot sound effect
```

### Table: `asset_cache`
| Column | Type | Description |
|---|---|---|
| `id` | UUID (PK) | Cache entry ID |
| `description_hash` | TEXT (UNIQUE) | SHA256 of the VLM sprite description |
| `file_path` | TEXT | Path to cached generated sprite |
| `created_at` | DATETIME | When cached |

> **Cache purpose:** If two trailers produce the same character description, skip regeneration. Saves minutes of SD inference.

---

## 7. ML Pipeline (The Brain)

### Step 1: Ingestion — ffmpeg Extraction
```python
# ingestion.py
# Extract 1 keyframe every 3 seconds (≈ 10-20 frames for a 2-minute trailer)
# Extract a 30-second audio clip from the middle (captures main theme)

ffmpeg -i trailer.mp4 -vf "fps=1/3,scale=512:288" keyframes/frame_%04d.png
ffmpeg -i trailer.mp4 -ss 00:00:30 -t 30 -vn -ar 16000 audio_sample.wav
```
Output: `/outputs/{id}/keyframes/` + `/outputs/{id}/audio_sample.wav`

### Step 2: VLM Analysis — The Game DNA
**Model:** GPT-4o mini via API (cheapest option; ~$0.01 per trailer)  
**Input:** 5 best keyframes (selected by frame entropy/sharpness score) + system prompt  
**Output:** Strict JSON — `game_dna.json`

#### VLM System Prompt
```
You are a video game analyst and pixel art director.
You will be shown keyframes from a modern game trailer.
Your job is to extract the game's DNA for an 8-bit demake.

You MUST respond with ONLY valid JSON matching this exact schema.
No explanation, no markdown, no preamble. JSON only.

{
  "title_guess": "string — your best guess at the game title",
  "genre": "one of: top_down_action_rpg | wave_shooter | open_world_sandbox | side_scroll_platformer",
  "setting": "string — one sentence describing the world (e.g. 'post-apocalyptic city')",
  "color_palette": ["#hex1", "#hex2", "#hex3", "#hex4"],  // 4-6 dominant colors
  "player_description": "string — describe the main character for pixel art generation",
  "enemy_description": "string — describe the main enemy type",
  "boss_description": "string — describe the boss or main antagonist",
  "environment_description": "string — describe the main environment/level",
  "music_vibe": "one of: intense_action | dark_horror | epic_adventure | urban_gritty | mysterious",
  "music_tempo": "one of: slow | medium | fast | frantic",
  "confidence": 0.0 to 1.0
}
```

#### Game DNA Example Output (CoD Zombies-style)
```json
{
  "title_guess": "Zombie Survival",
  "genre": "wave_shooter",
  "setting": "abandoned military bunker overrun with undead",
  "color_palette": ["#1a0a00", "#3d2b1f", "#8b4513", "#c8a882"],
  "player_description": "armored soldier in military fatigues, helmet, holding rifle, side view",
  "enemy_description": "shambling zombie in torn clothes, outstretched arms, rotting flesh",
  "boss_description": "massive zombie brute, oversized fists, chains, glowing eyes",
  "environment_description": "dark concrete bunker walls, metal floor grates, red emergency lighting",
  "music_vibe": "dark_horror",
  "music_tempo": "frantic",
  "confidence": 0.87
}
```

### Step 3: Genre Matching
```python
# genre_matcher.py
# Maps VLM genre string → template JSON file
# Falls back to "side_scroll_platformer" if confidence < 0.5

GENRE_MAP = {
    "top_down_action_rpg": "templates/top_down_action_rpg.json",
    "wave_shooter":        "templates/wave_shooter.json",
    "open_world_sandbox":  "templates/open_world_sandbox.json",
    "side_scroll_platformer": "templates/side_scroll_platformer.json"
}
```

### Step 4: Pydantic Validation
```python
class GameDNA(BaseModel):
    title_guess: str
    genre: Literal["top_down_action_rpg","wave_shooter","open_world_sandbox","side_scroll_platformer"]
    setting: str
    color_palette: list[str] = Field(min_length=4, max_length=6)
    player_description: str
    enemy_description: str
    boss_description: str
    environment_description: str
    music_vibe: Literal["intense_action","dark_horror","epic_adventure","urban_gritty","mysterious"]
    music_tempo: Literal["slow","medium","fast","frantic"]
    confidence: float = Field(ge=0.0, le=1.0)
```

---

## 8. Asset Generation Pipeline

### Stable Diffusion Setup
- **Model:** SD 1.5 with LCM (Latent Consistency Model) LoRA  
- **VRAM usage:** ~3.2GB at INT8 / bfloat16  
- **Steps:** 4-8 (LCM allows ultra-fast inference)  
- **Library:** `diffusers` + `torch`

### Pixel Art LoRA
Use a pre-trained pixel art LoRA (e.g., `pixel-art-xl` from CivitAI) to constrain output to NES-style sprites.

### Sprite Generation Prompts
Prompts are assembled from `game_dna.json` fields:

```python
def build_sprite_prompt(description: str, palette: list[str], slot: str) -> str:
    palette_str = ", ".join(palette)
    return f"""
pixel art spritesheet, NES 8-bit style, {description},
transparent background, white padding between frames,
4-frame animation, 16x16 pixels per frame,
color palette: {palette_str},
no anti-aliasing, hard edges, retro game sprite,
<lora:pixel_art_nes:0.9>
"""

NEGATIVE_PROMPT = """
photorealistic, 3D render, blurry, smooth gradients,
modern graphics, anti-aliased, watermark, text
"""
```

### Sprite Output Specs
| Slot | Canvas Size | Frame Count | Notes |
|---|---|---|---|
| `sprite_player` | 128x32 | 8 | idle(2) + walk(4) + attack(2) |
| `sprite_enemy` | 64x16 | 4 | walk(2) + attack(2) |
| `sprite_boss` | 256x64 | 6 | idle(2) + walk(2) + attack(2) |
| `sprite_projectile` | 32x8 | 2 | fly(2) |
| `tile_floor` | 16x16 | 1 | tileable |
| `tile_wall` | 16x16 | 1 | tileable |

### Asset Cache Check (Before Generation)
```python
import hashlib

async def get_or_generate_sprite(description: str, ...) -> str:
    hash_key = hashlib.sha256(description.encode()).hexdigest()
    cached = db.query(AssetCache).filter_by(description_hash=hash_key).first()
    if cached:
        return cached.file_path   # Skip generation entirely
    
    file_path = await run_stable_diffusion(description, ...)
    db.add(AssetCache(description_hash=hash_key, file_path=file_path))
    return file_path
```

---

## 9. Audio Pipeline

### Approach: Algorithmic MIDI → Web Chiptune
Zero VRAM. Pure Python math. Played in-browser via Tone.js with NES-style synth voices.

### NES Channel Constraints
```
Channel 1: Pulse Wave 1  — Melody / lead
Channel 2: Pulse Wave 2  — Harmony / counter-melody  
Channel 3: Triangle Wave — Bass line
Channel 4: Noise Channel — Percussion (hi-hat, snare, kick)
```

### Music Vibe → MIDI Parameters
```python
VIBE_CONFIGS = {
    "dark_horror": {
        "scale": "minor",
        "root_note": "A2",
        "tempo_bpm": 85,
        "melody_pattern": "descending_chromatic",
        "bass_pattern": "pulse_drone",
        "percussion_density": 0.4
    },
    "intense_action": {
        "scale": "minor_pentatonic",
        "root_note": "E2",
        "tempo_bpm": 160,
        "melody_pattern": "rapid_arpeggio",
        "bass_pattern": "driving_eighth",
        "percussion_density": 0.9
    },
    "epic_adventure": {
        "scale": "major",
        "root_note": "C3",
        "tempo_bpm": 120,
        "melody_pattern": "heroic_fanfare",
        "bass_pattern": "march_bass",
        "percussion_density": 0.6
    },
    "urban_gritty": {
        "scale": "blues",
        "root_note": "G2",
        "tempo_bpm": 95,
        "melody_pattern": "syncopated",
        "bass_pattern": "walking_bass",
        "percussion_density": 0.7
    }
}
```

### Audio Generation Flow
```
game_dna["music_vibe"] + game_dna["music_tempo"]
    → VIBE_CONFIGS lookup
    → Build 4-bar looping pattern per NES channel
    → Output: audio/bgm.mid
    → Browser loads bgm.mid → Tone.js plays with NES synth presets
```

### Tone.js NES Synth Config (frontend)
```javascript
const nesSynths = {
  pulse1: new Tone.Synth({ oscillator: { type: "square" }, volume: -6 }),
  pulse2: new Tone.Synth({ oscillator: { type: "square8" }, volume: -9 }),
  triangle: new Tone.Synth({ oscillator: { type: "triangle" }, volume: -3 }),
  noise: new Tone.NoiseSynth({ noise: { type: "white" }, volume: -12 })
};
```

---

## 10. Frontend Game Engine

### Framework: Phaser.js 3.x
Phaser handles: canvas rendering, physics, input, tilemaps, spritesheets, scene management.

### Page Flow
```
index.html          → Upload form + progress bar + WebSocket listener
    ↓ (status = "ready")
game.html?id={uuid} → Fetches manifest → boots Phaser
    ↓
PreloadScene        → Loads all sprites/audio from manifest paths
    ↓
GameScene           → Instantiates the correct template JS class
    ↓
HUDScene            → Overlays HP, score, ammo etc.
```

### Template Class Architecture
```javascript
// Each template extends a base class
class DemakeTemplate {
    constructor(scene, manifest) {
        this.scene = scene;
        this.manifest = manifest;
        this.config = manifest.game_config;
    }
    
    create() { /* Override: spawn player, enemies, world */ }
    update(time, delta) { /* Override: game loop logic */ }
    applyPalette() { /* Tint sprites to match extracted palette */ }
}

class WaveShooter extends DemakeTemplate {
    create() {
        this.spawnPlayer();
        this.startWave(1);
        this.setupHUD();
    }
    // ... wave logic, enemy spawning, collision, ammo system
}
```

### Color Palette Application
The VLM-extracted palette is applied as Phaser tint filters on all sprites, ensuring the game feels visually consistent with the source material even when sprites are generic.

```javascript
// Apply palette to player sprite
this.player.setTint(parseInt(manifest.palette.primary.replace('#', '0x')));
```

---

## 11. The JSON Manifest Contract

This is the central data structure that connects the ML pipeline to the game engine. Once this schema is stable, both sides can be developed independently.

```json
{
  "demake_id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "ZOMBIE BUNKER",
  "generated_at": "2024-01-15T14:23:00Z",
  "source_game_guess": "Zombie Survival",
  "template": "wave_shooter",
  
  "palette": {
    "primary":    "#1a0a00",
    "secondary":  "#3d2b1f",
    "accent":     "#8b4513",
    "highlight":  "#c8a882",
    "background": "#0d0500"
  },
  
  "game_config": {
    "physics": { "gravity": 0, "player_speed": 120, "bullet_speed": 300 },
    "hud": ["hp_bar", "ammo_counter", "wave_number", "score"],
    "waves": { "start_count": 5, "multiplier": 1.3, "boss_every": 5 },
    "player": { "max_hp": 100, "max_ammo": 30, "reload_time_ms": 1500 }
  },
  
  "assets": {
    "sprites": {
      "player":      "/api/v1/demake/{id}/asset/sprite_player.png",
      "enemy":       "/api/v1/demake/{id}/asset/sprite_enemy.png",
      "boss":        "/api/v1/demake/{id}/asset/sprite_boss.png",
      "projectile":  "/api/v1/demake/{id}/asset/sprite_projectile.png",
      "tile_floor":  "/api/v1/demake/{id}/asset/tile_floor.png",
      "tile_wall":   "/api/v1/demake/{id}/asset/tile_wall.png"
    },
    "sprite_meta": {
      "player": {
        "frame_width": 16, "frame_count": 8,
        "animations": {
          "idle":   { "frames": [0,1], "fps": 4, "loop": true },
          "walk":   { "frames": [2,3,4,5], "fps": 8, "loop": true },
          "attack": { "frames": [6,7], "fps": 12, "loop": false }
        }
      }
    },
    "audio": {
      "bgm": "/api/v1/demake/{id}/asset/bgm.mid",
      "sfx_shoot": "/api/v1/demake/{id}/asset/sfx_shoot.mid",
      "sfx_hit":   "/api/v1/demake/{id}/asset/sfx_hit.mid"
    }
  },
  
  "debug": {
    "vlm_confidence": 0.87,
    "generation_time_seconds": 142,
    "vram_peak_mb": 3840
  }
}
```

---

## 12. Sprint Plan

### Sprint 0: Genre Template Library *(~3 days)* — ✅ COMPLETE
**Goal:** A manually-authored, playable browser game for each template, driven by hardcoded JSON.

- [ ] Write 4 template JSON config files
- [ ] Implement Phaser.js base engine + `DemakeTemplate` class
- [ ] Implement all 4 template JS classes (WaveShooter, TopDownRPG, OpenWorld, Platformer)
- [ ] Build HUD scene with all HUD element types
- [ ] **Deliverable:** Visit `game.html?manifest=test_wave_shooter.json` and play a working (if ugly) game

### Sprint 1: Backend Scaffolding *(~4 days)* — ✅ COMPLETE
**Goal:** API endpoints live, SQLite wired, upload works, queue runs.

- [ ] FastAPI app setup with SQLAlchemy + WAL mode SQLite
- [ ] All 4 API endpoints + WebSocket
- [ ] File upload + validation (MP4 only, 100MB max)
- [ ] Async job queue (asyncio or Celery)
- [ ] Pipeline stub (each stage logs "stage X complete", sleeps 2s)
- [ ] **Deliverable:** Upload a video, poll status, see it move through fake stages, fetch a hardcoded manifest

### Sprint 2: Ingestion & VLM Analysis *(~5 days)* — ✅ COMPLETE
**Goal:** Upload a real trailer, get real `game_dna.json` back.

- [ ] ffmpeg keyframe + audio extraction
- [ ] Frame quality scoring (pick best 5)
- [ ] GPT-4o mini API integration
- [ ] Pydantic schema + validation + retry loop
- [ ] Genre matcher + fallback logic
- [ ] **Deliverable:** Upload a GTA or CoD trailer → receive valid `game_dna.json` in `/outputs/{id}/`

### Sprint 3: Asset & Audio Generation *(~7 days)* — ✅ COMPLETE
**Goal:** Automated sprites + MIDI from game DNA.

- [ ] SD 1.5 + LCM LoRA setup (pixel art)
- [ ] Prompt builder from DNA fields
- [ ] Spritesheet generation for all 6 slot types
- [ ] Asset cache table + hash-based deduplication
- [ ] Algorithmic MIDI generator (all 4 NES channels)
- [ ] Vibe → MIDI parameter mapping
- [ ] **Deliverable:** `game_dna.json` in → `/outputs/{id}/sprites/` + `/outputs/{id}/audio/` fully populated

### Sprint 4: Integration & Upload UI *(~5 days)* — ✅ COMPLETE
**Goal:** Full end-to-end: upload → wait → play.

- [ ] Wire all pipeline steps into `orchestrator.py`
- [ ] Build `index.html` upload UI with progress bar (WebSocket-driven)
- [ ] Build `game.html` manifest loader
- [ ] Error handling + "failed" state UI
- [ ] VRAM monitor + CPU fallback trigger
- [ ] **Deliverable:** Upload a real trailer, watch progress bar, click "Play Your Demake"

### Sprint 5: Polish & GitHub Release *(~4 days)*
**Goal:** Presentable open source repo.

- [ ] `README.md` with GIF demo, install steps, example output
- [ ] `docker-compose.yml` for one-command setup
- [ ] Example outputs (3 pre-generated demakes) committed
- [ ] `config.yaml` documentation
- [ ] MIT License

Sprint 5:   Polish & Wiring Review ✅
Sprint 5.5: Prompt & Cache Tuning (CLIP truncation fix, smarter dedup) ✅
Sprint 6:   Fine-tuned Sprites ✅
Sprint 6.5: WebSocket Hardening ✅

Sprint 7:   Tilemap Generation ✅ — WFC generator (`backend/pipeline/tilemap_gen.py`) +
            `TilemapRenderer` in `game.html` verified generating & rendering playable
            maps for all 5 genre templates.

---

## Sprint 8 Series — Full Gameplay Loops

Status after the Sprint 7 tilemap verification pass (each template boots and plays on a
WFC-generated map). The "full loop" depth work is partially done per genre:

| Sprint | Scope | Status |
|---|---|---|
| 8A | Shared game systems (entities, inventory, projectiles) | ✅ SharedSystems/Inventory/ProjectileSystem wired into wave shooter + ARPG; chest→potion→heal loop verified |
| 8B | Wave Shooter full loop | ✅ Waves, boss every N, ammo/reload, score, death |
| 8C | Top Down ARPG full loop | 🔶 Melee + projectile magic + chest loot + room clear done; final-room boss deferred to completion items |
| 8D | Open World full loop | ✅ Police escalation per star, busted/arrest state, wanted decay, pistol combat, WASTED/BUSTED split |
| 8E | Turn Based full loop | ✅ Gym tile at route's end → gym leader boss battle → badge victory; boss loss returns to overworld |

### Sprint 8A: Shared Game Systems — ✅ COMPLETE

**Plan & Scope:** Extract the combat plumbing duplicated across templates into reusable,
shared systems inside `game.html`, then wire them into the scenes that need them.

- [x] `SharedSystems.applyDamage()` — common damage helper: HP decrement, red flash,
      blood particles, death callback (deduplicates 4 near-identical code paths)
- [x] `Inventory` class — slot-based item store (add/remove/has/count), scene-event driven
      (`inventoryChanged`) so any HUD can subscribe
- [x] `ProjectileSystem` class — generic projectile pool: `fire(x, y, angle, speed, opts)`,
      TTL, auto-destroy on wall collision, damage payload, works for player AND enemy shots
- [x] Shared textures (`bolt`, `potion`) generated procedurally alongside existing ones
- [x] Wire into WaveShooter: replace inline bullet group with `ProjectileSystem`
      (+ bullets now collide with walls)
- [x] Wire into TopDownRPG: magic fires a projectile instead of instant AoE;
      chests from WFC `item` spawns openable with SPACE → potion loot into Inventory;
      Q consumes potion to heal (uses `applyDamage` inverse path)
- [x] ARPG HUD shows potion count via `inventoryChanged`
- [x] `?template=` URL param on `game.html` — boots any genre scene directly from
      fallback manifest for instant per-genre smoke testing (no backend/upload needed;
      unlocked by the Genre-Only direction)
- [x] Headless smoke harness (`tools/smoke.js`, puppeteer-core + installed Edge) —
      all 5 templates verified: correct scene active, NO CONSOLE ERRORS.
      Fixed two latent boot bugs found by the harness: (1) fallback path booted
      Phaser before scene classes were evaluated (TDZ crash — `loadManifest()` now
      deferred to script end), (2) scenes crashed on integer palette values
      (`hexColor()` helper added)
- [x] **Deliverable:** ARPG run: find chest → get potion → drink it → clear room with
      projectile magic. Wave shooter unchanged behaviorally but on shared systems.
      *(Verified by user play-through)*

### Sprint 8D Completion: Wanted System Escalation — ✅ COMPLETE

**Plan & Scope:** Turn the minimal wanted-stars into a real GTA-style heat system.

- [x] `Police` group — distinct from civilians (blue tint), spawn at world edges
      when wanted rises: 1 cop per star, speed + damage scale with wanted level
- [x] Civilians no longer turn hostile — they always wander; police handle
      all enforcement (removes old "everyone chases you" behavior)
- [x] Police contact damage scales with wanted level; death still possible
- [x] **Busted state**: at 3★+, if 2+ cops stay within grab range for 1.5s →
      "BUSTED!" — lose half your cash, wanted resets to 0, cops disperse,
      player respawns at spawn point (arrest instead of death)
- [x] Wanted decay: 1 star drops every 20s without committing a crime (evade mechanic)
- [x] HUD: wanted stars flash while cops are actively chasing; transient
      "BUSTED! LOST HALF CASH" overlay on arrest
- [x] Genre-aware procedural fallback maps for smoke testing (no-tilemap path):
      city blocks with roads/buildings for open world, room dividers + pillars
      for ARPG (was a featureless box before)
- [x] Open world combat: SPACE fires a pistol (shared ProjectileSystem) instead of
      melee — shooting a cop also raises wanted; cops have HP scaling with wanted
- [x] Death screen now reads "WASTED!" — visually distinct from arrest ("BUSTED!")
- [x] Smoke test `open_world_sandbox` clean (correct scene, no console errors)
- [x] **Deliverable:** manual run: shoot NPC → 1★ cop chases → escalate to 3★ →
      get busted → respawn with half cash → evade until stars decay to 0
      *(Verified by user play-through)*

### Sprint 8E Completion: Gym Boss Endpoint — ✅ COMPLETE

**Plan & Scope:** Give the turn-based loop a real win condition: a gym boss at the
end of the overworld route, replacing "survive N encounters" as the goal.

**Backend (`tilemap_gen.py`):**
- [x] `gym` tile type in the turn_based_rpg tile set (passable, triggers boss battle
      on step; visually distinct red + "G" marker in the renderer)
- [x] Turn-based map generation places the gym at the path's destination
      (bottom-right), with the approach kept clear of trees

**Frontend (`game.html`):**
- [x] `TurnOverworldScene`: stepping on a gym tile triggers a boss battle
      (also a fallback gym position + tall-grass encounter zones in the
      no-tilemap smoke path, via a dedicated overworld fallback map)
- [x] `TurnBasedScene`: boss battle mode — enemy HP scaled (tuned to 3× after
      play-testing), larger sprite, "GYM LEADER BATTLE" label, 1.5× attack
- [x] Boss win → "VICTORY! GYM BADGE EARNED!" end screen (the run's win condition);
      boss loss (death or run) → return to overworld just south of the gym at 60% HP
      (no dead-end refresh screen, no instant re-trigger)
- [x] Smoke test `turn_based_rpg` clean
- [x] **Deliverable:** manual run: fight encounters to level up → reach gym →
      beat the gym leader → victory screen *(verified by user play-through)*

### Sprint 8B–8E Completion Items

- [x] 8D: wanted-star escalation (police NPC spawn rate/damage scales per star),
      busted state when surrounded at 3+ stars (arrest instead of death)
- [x] 8E: gym/boss arena tile type in WFC tileset → final boss battle as win condition
      (replaces "N encounters" counter)
- [ ] 8C: boss enemy in final dungeon room (leverages shared systems from 8A)

### Sprint 8F: Genre-Only Mode ("Mode -1") — NEW

**Goal:** Generate a small, fully-fleshed playable world slice from just a genre pick.
No trailer upload, no VLM call, no video processing — instant demakes.

**Backend:**
- [ ] New endpoint `POST /api/v1/demake/generate` accepting `{ "genre": "<template_id>" }`
- [ ] DNA synthesizer: builds `game_dna.json` from the genre template defaults +
      seeded random variation (palette hue-shift, music vibe pick, setting flavor text)
      so repeated runs of the same genre don't look identical
- [ ] Orchestrator skips ingestion + VLM stages for these jobs (stage labels adjusted)

**Per-genre slice targets (small but complete):**
- [ ] Action RPG: multi-room dungeon with chests, enemy spawns, exit endpoint
- [ ] Wave Shooter: bunker arena with cover + ammo caches (already close — verify)
- [ ] Open World: city block with a few streets, NPCs, mission marker
- [ ] Platformer: one complete level ending in a goal flag (already close — verify)
- [ ] Turn-Based: overworld route with encounter zones + endpoint

**Frontend:**
- [ ] `index.html`: two entry points — **Genre-Only (default)** dropdown + Generate
      button; Upload Trailer moved to secondary tab
- [ ] WebSocket progress still streams for sprite/audio/tilemap stages

### Sprint 9: Multi-Trailer Blending *(trailer mode only)* ⬜
Feed two trailers → crossover demake ("What if Kingdom Hearts met CoD Zombies?").
Scoped to trailer mode; genre-only mode is unaffected.

### Sprint 9a: Godot Export ⬜ *(replaces the old "Unity/Godot" item — Godot chosen)*

**Decision:** Unity is out. **Godot 4** is the export target (already installed locally).

**Deliverable:** a manifest → Godot project generator. Phaser remains the live browser
player; Godot is an additional downloadable artifact per demake.

- [ ] New backend module `backend/pipeline/godot_export.py`: given a manifest, emit a
      complete Godot 4 project folder:
      - `project.godot` (viewport 480×320, pixel-art texture settings, input map)
      - `main.gd` + one GDScript scene script per genre template, mirroring the
        Phaser template classes (WaveShooter.tscn/gd, TopDownRPG, OpenWorld,
        Platformer, TurnBased)
      - Sprites copied into `assets/` with `.import` hints; tilemap grid →
        Godot `TileMapLayer` node data or generated `TileSet` resource
      - Audio: MIDI → converted OGG/WAV at export time (Godot has no native MIDI playback)
      - Spawn points from the WFC tilemap map to node placement in each scene
- [ ] API: `GET /api/v1/demake/{id}/godot` → zipped project download
- [ ] Verify round-trip: open generated project in Godot editor, press Play,
      gameplay matches the browser version for at least wave_shooter + top_down_action_rpg

### Sprint 10: IGDB Title Lookup ⬜
Known game titles skip VLM analysis entirely — look up cover art/screenshots/metadata
from IGDB and derive DNA deterministically. Works for both modes.

### Sprint 11: Save & Share ⬜
Hosted demakes with permanent URLs.

---

## 13. Tech Stack Summary

| Layer | Technology | Version | Notes |
|---|---|---|---|
| Backend | Python | 3.11+ | |
| Web Framework | FastAPI | 0.110+ | Async + WebSocket |
| ORM | SQLAlchemy | 2.0 | Async session |
| Database | SQLite | 3.x | WAL mode |
| Job Queue | asyncio / Celery | — | Start with asyncio, upgrade if needed |
| Video Processing | ffmpeg | 6.x | Via `ffmpeg-python` wrapper |
| VLM Analysis | GPT-4o mini | API | Cloud only, ~$0.01/trailer |
| Image Generation | Stable Diffusion 1.5 | diffusers | + LCM LoRA + pixel art LoRA |
| LLM (local) | Llama-3-8B | llama.cpp | 4-bit quantized, CPU fallback |
| ML Library | PyTorch | 2.2+ | CUDA + CPU |
| Frontend Framework | Phaser.js | 3.60+ | Canvas-based game engine |
| Audio Playback | Tone.js | 14.x | NES synth voices |
| Schema Validation | Pydantic | 2.x | Strict JSON contracts |
| Config | PyYAML | — | `config.yaml` runtime settings |

---

## 14. Future Roadmap

These are intentionally out of scope for v1 but worth designing toward:

- ~~v2: Tilemap generation~~ — **DONE in v1 (Sprint 7)**: WFC produces full tile grids with spawn points
- **v2: Fine-tuned NES sprite model** — Fine-tune SD on actual NES game spritesheets for authentic output (a real ML contribution)
- **v2: Genre-Only world options** — biome/size/palette pickers on top of Mode -1
- ~~v3: Multi-trailer blending~~ — promoted to Sprint 9 (trailer mode only)
- ~~v3: Save & share~~ — promoted to Sprint 11
- **v3: Godot web exports** — hosted Godot-exported demakes alongside Phaser builds (builds on Sprint 9a)
- **v4: Mobile export** — Package generated manifest as a React Native game

---

*Built for fun. Inspired by Kingdom Hearts, CoD Zombies, and GTA.*  
*Open source under MIT License.*

> **Changelog:** v1.1 — Two-mode entry (Genre-Only default / Trailer secondary), Sprint 8F added,
> Sprint 9a locked to Godot 4 export, sprint statuses reconciled with codebase state.
