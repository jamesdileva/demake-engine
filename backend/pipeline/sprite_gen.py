"""
Sprint 3 â€” Sprite Generation Pipeline

Uses Stable Diffusion 1.5 + LCM LoRA + pixel art LoRA to generate
NES-style sprite sheets from the game DNA descriptions.

Architecture doc reference:
  Run a highly quantized local Stable Diffusion model (SD 1.5 LCM with
  8-bit precision) fine-tuned on pixel art/spritesheets.
  Never load sprite model and LLM simultaneously â€” pipeline is sequential.

VRAM strategy:
  - SD 1.5 at bfloat16 â‰ˆ 3.2GB VRAM
  - LCM LoRA: 4-8 steps instead of 50 (much faster)
  - Unload model from VRAM after all sprites are done
  - CPU fallback if VRAM insufficient
"""
import os
import json
import hashlib
import shutil
from pathlib import Path
from PIL import Image, ImageDraw

from pipeline.validator import GameDNA
from database.db import SessionLocal
from database.models import Asset, AssetCache


# â”€â”€ Sprite slot definitions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Each slot defines the canvas size and animation frames to generate.
# Matches the architecture doc sprite output specs exactly.

SPRITE_SLOTS = {
    "sprite_player": {
        "canvas_w": 128, "canvas_h": 16,
        "frame_w":  16,  "frames": 8,
        "states":   ["idle", "idle", "walk", "walk", "walk", "walk", "shoot", "shoot"],
        "gen_size": 256,  # hero gets top quality; post-processed to 16px frames
    },
    "sprite_enemy": {
        "canvas_w": 64, "canvas_h": 16,
        "frame_w":  16, "frames": 4,
        "states":   ["walk", "walk", "attack", "attack"],
        "gen_size": 192,
    },
    "sprite_boss": {
        "canvas_w": 256, "canvas_h": 32,
        "frame_w":  32,  "frames": 6,
        "states":   ["idle", "idle", "walk", "walk", "attack", "attack"],
        "gen_size": 256,
    },
    "sprite_projectile": {
        "canvas_w": 32, "canvas_h": 8,
        "frame_w":  8,  "frames": 2,
        "states":   ["fly", "fly"],
        "gen_size": 128,
    },
    "tile_floor": {
        "canvas_w": 16, "canvas_h": 16,
        "frame_w":  16, "frames": 1,
        "states":   ["default"],
        "gen_size": 128,
    },
    "tile_wall": {
        "canvas_w": 16, "canvas_h": 16,
        "frame_w":  16, "frames": 1,
        "states":   ["default"],
        "gen_size": 128,
    },
}

NEGATIVE_PROMPT = (
    "photorealistic, 3D render, blurry, smooth gradients, modern graphics, "
    "anti-aliased, watermark, text, signature, username, high resolution, "
    "realistic lighting, shadows, depth of field"
)


# â”€â”€ Main entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_sprite_gen(dna: GameDNA, output_dir: str, config: dict) -> dict[str, str]:
    """
    Generate all sprite assets for a demake from the game DNA.

    Args:
        dna:        Validated GameDNA from VLM analysis
        output_dir: /outputs/{demake_id}/ â€” sprites written to sprites/ subdir
        config:     Loaded config.yaml

    Returns:
        Dict mapping slot_name â†’ file_path for all generated sprites
    """
    sprites_dir = os.path.normpath(os.path.join(output_dir, "sprites"))
    os.makedirs(sprites_dir, exist_ok=True)

    backend = _detect_backend(config)

    if backend == "procedural":
        print("[SpriteGen] No SD backend available â€” using procedural sprites")
        results = _generate_procedural(dna, sprites_dir)
    else:
        try:
            print(f"[SpriteGen] Starting generation with backend: {backend}")
            results = _generate_with_sd(dna, sprites_dir, config, backend)
        except Exception as e:
            print(f"[SpriteGen] {backend} failed ({e.__class__.__name__}: {e})")
            print("[SpriteGen] Falling back to procedural sprites")
            results = _generate_procedural(dna, sprites_dir)

    print(f"[SpriteGen] Generated {len(results)} sprite assets")
    return results


# â”€â”€ Backend detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _detect_backend(config: dict) -> str:
    """
    Detect the best available SD backend for this machine.

    Priority order:
      1. "cuda"      â€” NVIDIA GPU with CUDA (fastest)
      2. "directml"  â€” AMD/Intel GPU via ONNX DirectML (your RX 6400)
      3. "cpu"       â€” CPU only via ONNX (slowest, always works)
      4. "procedural"â€” No SD at all, PIL fallback

    Can be overridden in config.yaml:
      hardware:
        force_backend: "directml"   # or "cpu" or "procedural"
    """
    forced = config.get("hardware", {}).get("force_backend", "")
    if forced:
        print(f"[SpriteGen] Backend forced by config: {forced}")
        return forced

    if config.get("hardware", {}).get("force_cpu", False):
        return "cpu"

    try:
        import torch
        if torch.cuda.is_available():
            print("[SpriteGen] NVIDIA CUDA detected")
            return "cuda"
    except ImportError:
        pass

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "DmlExecutionProvider" in providers:
            print(f"[SpriteGen] AMD/Intel DirectML detected")
            return "directml"
        print(f"[SpriteGen] ONNX available, providers: {providers}")
        return "cpu"
    except ImportError:
        pass

    try:
        import diffusers
        return "cpu"
    except ImportError:
        pass

    return "procedural"


def _generate_with_sd(dna: GameDNA, sprites_dir: str,
                      config: dict, backend: str) -> dict[str, str]:
    """
    Dispatch to the correct SD backend based on detected hardware.

    backend = "cuda"      â†’ standard diffusers pipeline on NVIDIA
    backend = "directml"  â†’ ONNX pipeline on AMD/Intel via DirectML
    backend = "cpu"       â†’ ONNX pipeline on CPU (slow but works)
    """
    if backend == "cuda":
        return _generate_cuda(dna, sprites_dir)
    elif backend in ("directml", "cpu"):
        return _generate_onnx(dna, sprites_dir, backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# â”€â”€ CUDA backend (NVIDIA only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _generate_cuda(dna: GameDNA, sprites_dir: str) -> dict[str, str]:
    """Standard diffusers pipeline for NVIDIA GPUs."""
    import torch
    from diffusers import StableDiffusionPipeline, LCMScheduler

    print("[SpriteGen] Loading SD 1.5 on CUDA")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.bfloat16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    results = _run_generation_loop(pipe, dna, sprites_dir,
                                   steps=6, guidance=1.5,
                                   clear_fn=lambda: torch.cuda.empty_cache())
    del pipe
    torch.cuda.empty_cache()
    print("[SpriteGen] CUDA model unloaded")
    return results


# â”€â”€ ONNX / DirectML backend (AMD RX 6400 + Intel) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _generate_onnx(dna: GameDNA, sprites_dir: str, backend: str) -> dict[str, str]:
    """
    ONNX Runtime pipeline â€” works on AMD GPUs via DirectML on Windows.

    First run: exports SD 1.5 to ONNX format (~5 min one-time cost, ~3GB disk).
    Subsequent runs: loads cached ONNX model (~30-60s load time).

    Expected generation time per sprite on RX 6400:
      DirectML: 2-8 minutes per sprite
      CPU:      10-20 minutes per sprite
    """
    try:
        from optimum.onnxruntime import ORTStableDiffusionPipeline
    except ImportError:
        raise RuntimeError(
            "optimum not installed. Run: pip install optimum[onnxruntime-directml]"
        )

    import onnxruntime as ort

    # DirectML = AMD/Intel GPU acceleration on Windows
    # CPUExecutionProvider = CPU fallback
    provider = "DmlExecutionProvider" if backend == "directml" else "CPUExecutionProvider"

    # ONNX model cache directory â€” exported once, reused every run
    onnx_cache = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "models", "sd15_onnx")
    )
    os.makedirs(onnx_cache, exist_ok=True)

    unet_path = os.path.join(onnx_cache, "unet", "model.onnx")

    if not os.path.exists(unet_path):
        print(f"[SpriteGen] First run â€” exporting SD 1.5 to ONNX format")
        print(f"[SpriteGen] This takes ~5 minutes and ~3GB disk space. Only happens once.")
        print(f"[SpriteGen] Saving to: {onnx_cache}")
        pipe = ORTStableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            export=True,
            provider=provider,
        )
        pipe.save_pretrained(onnx_cache)
        print(f"[SpriteGen] ONNX export complete")
    else:
        print(f"[SpriteGen] Loading cached ONNX model from {onnx_cache}")
        pipe = ORTStableDiffusionPipeline.from_pretrained(
            onnx_cache,
            provider=provider,
        )

    print(f"[SpriteGen] Running on: {provider}")
    print(f"[SpriteGen] Note: DirectML/ONNX is slower than CUDA â€” expect 2-8 min per sprite")

    # ONNX pipeline uses more inference steps than LCM (no LCM LoRA support)
    # 20 steps is a good quality/speed balance for ONNX
    results = _run_generation_loop(pipe, dna, sprites_dir,
                                   steps=20, guidance=7.5,
                                   clear_fn=lambda: None)
    del pipe
    print("[SpriteGen] ONNX model unloaded")
    return results


# â”€â”€ Shared generation loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _run_generation_loop(pipe, dna: GameDNA, sprites_dir: str,
                         steps: int, guidance: float,
                         clear_fn) -> dict[str, str]:
    """
    Shared sprite generation loop used by both CUDA and ONNX backends.
    Iterates over all sprite slots, checks cache, generates, saves.
    """
    results = {}
    palette_str = ", ".join(dna.color_palette[:4])
    total = len(SPRITE_SLOTS)

    # Projectile only relevant for shooter templates
    NO_PROJECTILE_TEMPLATES = {"turn_based_rpg", "top_down_action_rpg"}
    template = dna.genre

    for i, (slot_name, slot_cfg) in enumerate(SPRITE_SLOTS.items(), 1):
        if slot_name == "sprite_projectile" and template in NO_PROJECTILE_TEMPLATES:
            print(f"[SpriteGen] ({i}/{total}) Skipping projectile â€” not needed for {template}")
            continue
        prompt    = _build_prompt(slot_name, dna, palette_str, slot_cfg)
        # Include title + pipeline version in cache key â€” prevents cross-game
        # sprite reuse AND stale pre-refactor sprites (v2 = 512px + postprocess)
        cache_key = hashlib.sha256(f"v2:{dna.title_guess}:{prompt}".encode()).hexdigest()

        cached_path = _check_cache(cache_key)
        if cached_path:
            print(f"[SpriteGen] ({i}/{total}) Cache hit: {slot_name}")
            out_path = os.path.normpath(os.path.join(sprites_dir, f"{slot_name}.png"))
            shutil.copy2(cached_path, out_path)
            results[slot_name] = out_path
            continue

        print(f"[SpriteGen] ({i}/{total}) Generating: {slot_name} "
              f"(steps={steps}, size={slot_cfg['gen_size']}px)...")

        gen_size = slot_cfg["gen_size"]
        image = pipe(
            prompt          = prompt,
            negative_prompt = NEGATIVE_PROMPT,
            width           = gen_size,
            height          = gen_size,
            num_inference_steps = steps,
            guidance_scale  = guidance,
        ).images[0]

        sheet    = _build_spritesheet(image, slot_cfg)
        out_path = os.path.normpath(os.path.join(sprites_dir, f"{slot_name}.png"))
        sheet.save(out_path, "PNG")
        results[slot_name] = out_path

        _write_cache(cache_key, out_path)
        clear_fn()
        print(f"[SpriteGen] ({i}/{total}) [OK] Saved: {out_path}")

    return results


def _build_prompt(slot_name: str, dna: GameDNA,
                  palette_str: str, slot_cfg: dict) -> str:
    """
    Build a sprite-specific SD prompt within CLIP's 77 token limit.
    Shorter prompts = better SD output. Keep descriptions under 50 chars.
    """
    # Truncate description to ~40 chars to stay within token limit
    def short(desc: str, maxlen: int = 40) -> str:
        return desc[:maxlen].strip() if desc else ""

    base = f"pixel art NES 8-bit sprite, transparent background, {palette_str[:30]}, "

    descriptions = {
        "sprite_player":     f"{short(dna.player_description)}, walk animation, 8 frames",
        "sprite_enemy":      f"{short(dna.enemy_description)}, walk attack animation, 4 frames",
        "sprite_boss":       f"{short(dna.boss_description)}, boss sprite, large, 6 frames",
        "sprite_projectile": "magic bolt projectile, glowing, 2 frames",
        "tile_floor":        f"{short(dna.environment_description, 30)}, floor tile, tileable",
        "tile_wall":         f"{short(dna.environment_description, 30)}, wall tile, tileable",
    }

    negative = "photorealistic, 3D, blurry, anti-aliased, modern"
    prompt = base + descriptions.get(slot_name, "game sprite")
    return prompt


def _postprocess_sprite(image: Image.Image, frame_w: int, frame_h: int) -> Image.Image:
    """
    Clean a raw SD generation into a crisp, readable pixel-art frame.

    Steps (Sprint 9a audit):
      1. Smooth-resize to a working canvas (LANCZOS keeps shapes coherent)
      2. Background removal â€” border-color flood by distance threshold
         ("transparent background" prompts rarely work on local SD)
      3. Auto-crop to the sprite bounding box, center on a square canvas
      4. Crisp NEAREST downscale to the target frame size
      5. Median-cut palette quantization (NES-style limited palette)
    """
    WORK = 128
    img = image.convert("RGB").resize((WORK, WORK), Image.LANCZOS)
    px = img.load()

    # 2. Estimate background color from the four corners (median per channel)
    corners = [px[0, 0], px[WORK - 1, 0], px[0, WORK - 1], px[WORK - 1, WORK - 1]]
    bg = tuple(sorted(c[i] for c in corners)[1] for i in range(3))

    def _close(c1, c2, thresh=2400):
        return sum((a - b) ** 2 for a, b in zip(c1[:3], c2)) < thresh

    rgba = img.convert("RGBA")
    data = rgba.load()
    for y in range(WORK):
        for x in range(WORK):
            if _close(data[x, y], bg):
                data[x, y] = (0, 0, 0, 0)

    # 3. Crop to content, center on square
    bbox = rgba.getbbox()
    if bbox:
        rgba = rgba.crop(bbox)
    side = max(rgba.size)
    sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    sq.paste(rgba, ((side - rgba.width) // 2, (side - rgba.height) // 2))

    # 4. Crisp downscale to frame size
    small = sq.resize((frame_w, frame_h), Image.NEAREST)

    # 5. Palette quantization on visible pixels only
    rgb = small.convert("RGB")
    quantized = rgb.quantize(colors=10, method=Image.MEDIANCUT, dither=Image.NONE)
    out = quantized.convert("RGBA")
    out.putalpha(small.getchannel("A"))
    return out


def _build_spritesheet(image: Image.Image, slot_cfg: dict) -> Image.Image:
    """
    Take a single SD-generated image and turn it into a proper spritesheet.

    Sprint 9a audit fix â€” the old pipeline squashed the raw 64px SD output
    straight to 16px, baking generation noise into every frame (the
    "rainbow garbage" look). Now: background removal -> auto-crop -> center
    -> crisp downscale -> palette quantization -> tile with hue variation.
    """
    frame_w  = slot_cfg["frame_w"]
    canvas_h = slot_cfg["canvas_h"]
    frames   = slot_cfg["frames"]
    canvas_w = slot_cfg["canvas_w"]

    # Post-process the raw SD image into one clean, tiny frame
    frame_img = _postprocess_sprite(image, frame_w, canvas_h)

    # Build spritesheet
    sheet = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    for i in range(frames):
        # Apply slight brightness variation per frame to suggest animation
        variation = _vary_frame(frame_img, i, frames)
        sheet.paste(variation, (i * frame_w, 0))

    return sheet


def _vary_frame(img: Image.Image, frame_idx: int, total: int) -> Image.Image:
    """Apply subtle per-frame variation to fake animation cycles."""
    from PIL import ImageEnhance
    # Slight brightness pulse: frames oscillate between 90% and 110% brightness
    t = frame_idx / max(total - 1, 1)
    brightness = 0.92 + 0.16 * abs(t - 0.5) * 2   # 0.92 â†’ 1.08 â†’ 0.92
    return ImageEnhance.Brightness(img).enhance(brightness)


# â”€â”€ Asset cache helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _check_cache(description_hash: str) -> str | None:
    """Check if this sprite description was generated before."""
    db = SessionLocal()
    try:
        entry = db.query(AssetCache).filter_by(
            description_hash=description_hash
        ).first()
        if entry and os.path.exists(entry.file_path):
            return entry.file_path
        return None
    finally:
        db.close()


def _write_cache(description_hash: str, file_path: str):
    """Save a new cache entry."""
    db = SessionLocal()
    try:
        entry = AssetCache(description_hash=description_hash, file_path=file_path)
        db.merge(entry)
        db.commit()
    except Exception as e:
        print(f"[SpriteGen] Cache write failed (non-fatal): {e}")
    finally:
        db.close()


# â”€â”€ Procedural fallback â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# When SD isn't available, generate pixel art programmatically using PIL.
# These won't look as good but the game will still be fully playable.
# The game engine (Sprint 0) already does this â€” we match that style here.

def _generate_procedural(dna: GameDNA, sprites_dir: str) -> dict[str, str]:
    """
    Generate placeholder pixel art sprites using PIL when SD isn't available.
    Uses the extracted color palette so sprites still match the game's vibe.
    """
    results = {}
    palette = _parse_palette(dna.color_palette)
    primary, secondary, accent, highlight = palette[:4]

    for slot_name, slot_cfg in SPRITE_SLOTS.items():
        out_path = os.path.join(sprites_dir, f"{slot_name}.png")

        sheet = Image.new("RGBA",
                          (slot_cfg["canvas_w"], slot_cfg["canvas_h"]),
                          (0, 0, 0, 0))
        draw  = ImageDraw.Draw(sheet)

        for i in range(slot_cfg["frames"]):
            ox = i * slot_cfg["frame_w"]
            fw = slot_cfg["frame_w"]
            fh = slot_cfg["canvas_h"]
            _draw_sprite_frame(draw, slot_name, ox, fw, fh, i,
                               primary, secondary, accent, highlight)

        sheet.save(out_path, "PNG")
        results[slot_name] = out_path
        print(f"[SpriteGen] Procedural fallback: {slot_name}")

    return results


def _parse_palette(hex_colors: list[str]) -> list[tuple]:
    """Convert hex color strings to RGB tuples."""
    result = []
    defaults = [(26,10,0), (61,43,31), (139,26,0), (255,68,0)]
    for i, hex_c in enumerate(hex_colors[:4]):
        try:
            hex_c = hex_c.lstrip("#")
            result.append(tuple(int(hex_c[j:j+2], 16) for j in (0, 2, 4)))
        except Exception:
            result.append(defaults[i])
    while len(result) < 4:
        result.append(defaults[len(result)])
    return result


def _draw_sprite_frame(draw: ImageDraw.ImageDraw, slot_name: str,
                       ox: int, fw: int, fh: int, frame_idx: int,
                       primary, secondary, accent, highlight):
    """Draw a single sprite frame procedurally based on slot type."""
    bob = frame_idx % 2  # Simple animation bob

    if slot_name == "sprite_player":
        # Body
        draw.rectangle([ox+4, 6+bob, ox+fw-4, fh-3], fill=primary)
        # Head
        draw.rectangle([ox+5, 2+bob, ox+fw-5, 6+bob], fill=secondary)
        # Eye
        draw.rectangle([ox+6, 3+bob, ox+8, 4+bob], fill=highlight)
        # Gun
        draw.rectangle([ox+fw-4, 7+bob, ox+fw, 8+bob], fill=accent)

    elif slot_name == "sprite_enemy":
        # Body
        draw.rectangle([ox+3, 5+bob, ox+fw-3, fh-2], fill=accent)
        # Head
        draw.rectangle([ox+4, 2+bob, ox+fw-4, 5+bob], fill=secondary)
        # Red eyes
        draw.rectangle([ox+5, 3+bob, ox+7, 4+bob], fill=(255, 0, 0, 255))
        draw.rectangle([ox+fw-7, 3+bob, ox+fw-5, 4+bob], fill=(255, 0, 0, 255))

    elif slot_name == "sprite_boss":
        # Large body
        draw.rectangle([ox+4, fh//3+bob, ox+fw-4, fh-2], fill=accent)
        # Large head
        draw.rectangle([ox+2, 2+bob, ox+fw-2, fh//3+bob], fill=secondary)
        # Glowing eyes
        draw.rectangle([ox+4, 5+bob, ox+8, 8+bob], fill=(255, 100, 0, 255))
        draw.rectangle([ox+fw-8, 5+bob, ox+fw-4, 8+bob], fill=(255, 100, 0, 255))

    elif slot_name == "sprite_projectile":
        # Small glowing bullet
        draw.rectangle([ox, fh//2-1, ox+fw-2, fh//2+1], fill=highlight)
        draw.rectangle([ox, fh//2, ox+2, fh//2], fill=(255, 255, 255, 255))

    elif slot_name == "tile_floor":
        # Concrete tile with grid lines
        draw.rectangle([ox, 0, ox+fw-1, fh-1], fill=primary)
        draw.line([ox, fh//2, ox+fw, fh//2], fill=secondary, width=1)
        draw.line([ox+fw//2, 0, ox+fw//2, fh], fill=secondary, width=1)

    elif slot_name == "tile_wall":
        # Brick pattern
        draw.rectangle([ox, 0, ox+fw-1, fh-1], fill=secondary)
        draw.rectangle([ox+1, 1, ox+fw//2-1, fh//2-1], fill=primary)
        draw.rectangle([ox+fw//2+1, fh//2+1, ox+fw-2, fh-2], fill=primary)