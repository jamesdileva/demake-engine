"""
Sprint 3 — Chiptune Audio Generation

Algorithmically generates MIDI files constrained to NES-style channels.
Zero VRAM — pure Python math. Played in-browser via Tone.js.

Architecture doc reference:
  NES Channel Constraints:
    Channel 1: Pulse Wave 1  — Melody / lead
    Channel 2: Pulse Wave 2  — Harmony / counter-melody
    Channel 3: Triangle Wave — Bass line
    Channel 4: Noise Channel — Percussion

  Use a local LLM or algorithmic script to generate MIDI files
  constrained to NES-style channels, then play through a web-based
  chiptune synthesizer (Tone.js).
"""
import os
import random
from midiutil import MIDIFile
from pipeline.validator import GameDNA


# ── Music theory constants ─────────────────────────────────────────────────────

# MIDI note numbers — middle C = 60
NOTES = {
    "C": 60, "C#": 61, "D": 62, "D#": 63, "E": 64,
    "F": 65, "F#": 66, "G": 67, "G#": 68, "A": 69,
    "A#": 70, "B": 71,
}

# Scale intervals (semitones from root)
SCALES = {
    "major":           [0, 2, 4, 5, 7, 9, 11],
    "minor":           [0, 2, 3, 5, 7, 8, 10],
    "minor_pentatonic":[0, 3, 5, 7, 10],
    "major_pentatonic":[0, 2, 4, 7, 9],
    "blues":           [0, 3, 5, 6, 7, 10],
    "chromatic":       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}

# NES MIDI channels (0-indexed)
CH_PULSE1   = 0   # Melody
CH_PULSE2   = 1   # Harmony
CH_TRIANGLE = 2   # Bass
CH_NOISE    = 9   # Drums (channel 10 in MIDI = index 9, always drums)

# GM drum note numbers for noise channel
DRUM_KICK  = 36
DRUM_SNARE = 38
DRUM_HAT   = 42
DRUM_CRASH = 49


# ── Vibe configurations ───────────────────────────────────────────────────────
# Maps music_vibe + music_tempo → MIDI generation parameters
# Matches the architecture doc VIBE_CONFIGS table exactly.

VIBE_CONFIGS = {
    "dark_horror": {
        "scale":       "minor",
        "root_note":   "A",
        "root_octave": 3,
        "base_bpm":    75,
        "melody_pattern":    "descending_chromatic",
        "harmony_pattern":   "drone",
        "bass_pattern":      "pulse_drone",
        "percussion_density": 0.35,
        "melody_velocity":   70,
        "note_length":       0.75,   # Longer, eerie notes
    },
    "intense_action": {
        "scale":       "minor_pentatonic",
        "root_note":   "E",
        "root_octave": 3,
        "base_bpm":    155,
        "melody_pattern":    "rapid_arpeggio",
        "harmony_pattern":   "parallel_thirds",
        "bass_pattern":      "driving_eighth",
        "percussion_density": 0.9,
        "melody_velocity":   100,
        "note_length":       0.25,   # Short punchy notes
    },
    "epic_adventure": {
        "scale":       "major",
        "root_note":   "C",
        "root_octave": 4,
        "base_bpm":    118,
        "melody_pattern":    "heroic_fanfare",
        "harmony_pattern":   "parallel_thirds",
        "bass_pattern":      "march_bass",
        "percussion_density": 0.6,
        "melody_velocity":   90,
        "note_length":       0.5,
    },
    "urban_gritty": {
        "scale":       "blues",
        "root_note":   "G",
        "root_octave": 3,
        "base_bpm":    92,
        "melody_pattern":    "syncopated",
        "harmony_pattern":   "drone",
        "bass_pattern":      "walking_bass",
        "percussion_density": 0.7,
        "melody_velocity":   85,
        "note_length":       0.4,
    },
    "mysterious": {
        "scale":       "minor",
        "root_note":   "D",
        "root_octave": 3,
        "base_bpm":    88,
        "melody_pattern":    "sparse_arpeggios",
        "harmony_pattern":   "drone",
        "bass_pattern":      "pulse_drone",
        "percussion_density": 0.2,
        "melody_velocity":   65,
        "note_length":       1.0,
    },
}

# Tempo multipliers per music_tempo value
TEMPO_MULTIPLIERS = {
    "slow":    0.65,
    "medium":  0.85,
    "fast":    1.15,
    "frantic": 1.45,
}


# ── Main entry point ───────────────────────────────────────────────────────────

def run_audio_gen(dna: GameDNA, output_dir: str) -> dict[str, str]:
    """
    Generate all audio assets for a demake from the game DNA.

    Args:
        dna:        Validated GameDNA (uses music_vibe and music_tempo)
        output_dir: /outputs/{demake_id}/ — audio written to audio/ subdir

    Returns:
        Dict mapping track_name → file_path
    """
    audio_dir = os.path.normpath(os.path.join(output_dir, "audio"))
    os.makedirs(audio_dir, exist_ok=True)

    results = {}

    # ── Background music ──────────────────────────────────────────────────────
    bgm_path = os.path.join(audio_dir, "bgm.mid")
    _generate_bgm(dna, bgm_path)
    results["bgm"] = bgm_path

    # ── Sound effects ─────────────────────────────────────────────────────────
    sfx_shoot = os.path.join(audio_dir, "sfx_shoot.mid")
    _generate_sfx_shoot(sfx_shoot)
    results["sfx_shoot"] = sfx_shoot

    sfx_hit = os.path.join(audio_dir, "sfx_hit.mid")
    _generate_sfx_hit(sfx_hit)
    results["sfx_hit"] = sfx_hit

    sfx_wave = os.path.join(audio_dir, "sfx_wave_clear.mid")
    _generate_sfx_wave_clear(sfx_wave)
    results["sfx_wave_clear"] = sfx_wave

    print(f"[AudioGen] Generated {len(results)} audio tracks")
    for name, path in results.items():
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"[AudioGen]   {name}: {path} ({size} bytes)")

    return results


# ── BGM generator ─────────────────────────────────────────────────────────────

def _generate_bgm(dna: GameDNA, output_path: str):
    """
    Generate a 4-bar looping NES-style background music track.
    Uses all 4 NES channels: pulse1 (melody), pulse2 (harmony),
    triangle (bass), noise (drums).
    """
    vibe_key = dna.music_vibe
    cfg = VIBE_CONFIGS.get(vibe_key, VIBE_CONFIGS["intense_action"])

    # Apply tempo multiplier
    tempo_mult = TEMPO_MULTIPLIERS.get(dna.music_tempo, 1.0)
    bpm = int(cfg["base_bpm"] * tempo_mult)
    bpm = max(60, min(200, bpm))  # Clamp to sane range

    print(f"[AudioGen] BGM: vibe={vibe_key}, tempo={dna.music_tempo}, bpm={bpm}")

    # Build scale note list across 2 octaves
    root_midi = NOTES[cfg["root_note"]] + (cfg["root_octave"] - 4) * 12
    intervals = SCALES[cfg["scale"]]
    scale_notes = []
    for octave in range(3):
        for interval in intervals:
            note = root_midi + interval + (octave * 12)
            if 24 <= note <= 108:   # MIDI range safety
                scale_notes.append(note)

    # 4 tracks, 4 bars each, 4/4 time
    midi = MIDIFile(4)
    bars       = 4
    beats_per_bar = 4

    for track, ch in enumerate([CH_PULSE1, CH_PULSE2, CH_TRIANGLE, CH_NOISE]):
        midi.addTempo(track, 0, bpm)
        if ch != CH_NOISE:  # Don't set program on drum channel
            midi.addProgramChange(track, ch, 0, 80)

    # ── Channel 1: Melody (Pulse 1) ───────────────────────────────────────────
    melody_notes = _build_melody(
        cfg["melody_pattern"], scale_notes,
        bars, beats_per_bar, cfg["note_length"]
    )
    for beat, note, duration, velocity in melody_notes:
        midi.addNote(0, CH_PULSE1, note,
                     beat, duration, int(velocity * cfg["melody_velocity"] / 100))

    # ── Channel 2: Harmony (Pulse 2) ─────────────────────────────────────────
    harmony_notes = _build_harmony(
        cfg["harmony_pattern"], melody_notes, scale_notes
    )
    for beat, note, duration, velocity in harmony_notes:
        midi.addNote(1, CH_PULSE2, note,
                     beat, duration, int(velocity * 75 / 100))

    # ── Channel 3: Bass (Triangle) ────────────────────────────────────────────
    bass_notes = _build_bass(
        cfg["bass_pattern"], scale_notes[:7],
        bars, beats_per_bar, root_midi
    )
    for beat, note, duration, velocity in bass_notes:
        midi.addNote(2, CH_TRIANGLE, note, beat, duration, velocity)

    # ── Channel 4: Drums (Noise) ──────────────────────────────────────────────
    drum_events = _build_drums(
        cfg["percussion_density"], bars, beats_per_bar
    )
    for beat, drum_note, duration, velocity in drum_events:
        midi.addNote(3, CH_NOISE, drum_note, beat, duration, velocity)

    with open(output_path, "wb") as f:
        midi.writeFile(f)


# ── Melody patterns ────────────────────────────────────────────────────────────

def _build_melody(pattern: str, scale: list[int],
                  bars: int, bpb: int, note_len: float) -> list[tuple]:
    """Build melody notes as list of (beat, note, duration, velocity)."""
    events = []
    total_beats = bars * bpb
    beat = 0.0

    # Use upper half of scale for melody
    melody_scale = [n for n in scale if n >= scale[len(scale)//2]]
    if not melody_scale:
        melody_scale = scale[-8:]

    if pattern == "rapid_arpeggio":
        step = 0.25
        idx = 0
        while beat < total_beats:
            note = melody_scale[idx % len(melody_scale)]
            vel  = random.randint(85, 100) if idx % 4 == 0 else random.randint(70, 85)
            events.append((beat, note, step * 0.9, vel))
            beat += step
            idx += 1

    elif pattern == "descending_chromatic":
        # Slow descending pattern — eerie horror feel
        step = 0.75
        notes_desc = sorted(melody_scale, reverse=True)
        idx = 0
        while beat < total_beats:
            note = notes_desc[idx % len(notes_desc)]
            events.append((beat, note, step * 0.8, random.randint(60, 75)))
            beat += step
            idx += 1

    elif pattern == "heroic_fanfare":
        # Dotted quarter + eighth feel, ascending
        pattern_beats = [1.5, 0.5, 1.0, 1.0]
        pattern_idx   = [0, 2, 4, 3, 5, 4, 6, 5]  # Scale degrees
        beat_pos = 0.0
        pi = 0
        while beat_pos < total_beats:
            dur = pattern_beats[pi % len(pattern_beats)]
            note_idx = pattern_idx[pi % len(pattern_idx)]
            note = melody_scale[note_idx % len(melody_scale)]
            vel  = 95 if pi % len(pattern_beats) == 0 else 80
            events.append((beat_pos, note, dur * 0.85, vel))
            beat_pos += dur
            pi += 1

    elif pattern == "syncopated":
        # Off-beat rhythm with rests
        pattern_beats = [0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5]
        rests         = [False, True, False, False, True, False, False]
        pi = 0
        while beat < total_beats:
            dur = pattern_beats[pi % len(pattern_beats)]
            if not rests[pi % len(rests)]:
                note = random.choice(melody_scale[-5:])
                events.append((beat, note, dur * 0.8, random.randint(75, 90)))
            beat += dur
            pi += 1

    elif pattern == "sparse_arpeggios":
        # Long notes with gaps — mysterious
        arp = [melody_scale[i % len(melody_scale)] for i in [0, 2, 4, 2]]
        beat_pos = 0.0
        idx = 0
        while beat_pos < total_beats:
            note = arp[idx % len(arp)]
            events.append((beat_pos, note, 0.75, random.randint(55, 70)))
            beat_pos += 1.5   # Gap between notes
            idx += 1

    else:
        # Generic fallback — simple quarter notes up the scale
        idx = 0
        while beat < total_beats:
            note = melody_scale[idx % len(melody_scale)]
            events.append((beat, note, 0.9, 80))
            beat += 1.0
            idx += 1

    return events


# ── Harmony patterns ───────────────────────────────────────────────────────────

def _build_harmony(pattern: str, melody: list[tuple],
                   scale: list[int]) -> list[tuple]:
    """Build harmony by transforming the melody."""
    if not melody:
        return []

    if pattern == "parallel_thirds":
        # Harmony a third below melody
        harmony = []
        for beat, note, dur, vel in melody:
            # Find note 2 scale degrees below
            if note - 3 in scale:
                h_note = note - 3
            elif note - 4 in scale:
                h_note = note - 4
            else:
                h_note = note - 3
            harmony.append((beat, h_note, dur, int(vel * 0.8)))
        return harmony

    elif pattern == "drone":
        # Sustained root note — eerie or mysterious
        root = scale[0]
        return [(0.0, root, len(melody) * 0.5, 50)]   # One long held note

    return []


# ── Bass patterns ──────────────────────────────────────────────────────────────

def _build_bass(pattern: str, low_scale: list[int],
                bars: int, bpb: int, root: int) -> list[tuple]:
    """Build bass line — triangle channel stays low."""
    events  = []
    total   = bars * bpb
    # Bass plays 1-2 octaves below root
    bass_root  = max(24, root - 12)
    bass_fifth = bass_root + 7

    if pattern == "driving_eighth":
        beat = 0.0
        notes = [bass_root, bass_root, bass_fifth, bass_root]
        idx = 0
        while beat < total:
            note = notes[idx % len(notes)]
            events.append((beat, note, 0.4, 90))
            beat += 0.5
            idx += 1

    elif pattern == "march_bass":
        # Oom-pah feel
        beat = 0.0
        while beat < total:
            events.append((beat, bass_root, 0.9, 95))     # Downbeat
            if beat + 1 < total:
                events.append((beat + 1, bass_fifth, 0.4, 75))  # Upbeat
            if beat + 2 < total:
                events.append((beat + 2, bass_root, 0.4, 80))
            if beat + 3 < total:
                events.append((beat + 3, bass_fifth, 0.4, 70))
            beat += 4

    elif pattern == "walking_bass":
        # Jazz-style chromatic walk
        beat = 0.0
        walk = [bass_root, bass_root+2, bass_root+4, bass_root+5,
                bass_root+7, bass_root+5, bass_root+4, bass_root+2]
        idx = 0
        while beat < total:
            note = walk[idx % len(walk)]
            events.append((beat, note, 0.45, 85))
            beat += 0.5
            idx += 1

    elif pattern == "pulse_drone":
        # Repeated root pulses
        beat = 0.0
        while beat < total:
            events.append((beat, bass_root, 0.4, 80))
            beat += 1.0

    return events


# ── Drum patterns ──────────────────────────────────────────────────────────────

def _build_drums(density: float, bars: int, bpb: int) -> list[tuple]:
    """
    Build drum pattern for the noise channel.
    density: 0.0 = minimal, 1.0 = full on.
    """
    events = []
    total  = bars * bpb

    beat = 0.0
    while beat < total:
        bar_beat = beat % bpb

        # Kick on beats 1 and 3 (always, regardless of density)
        if bar_beat in [0, 2]:
            events.append((beat, DRUM_KICK, 0.4, 100))

        # Snare on beats 2 and 4 (if density > 0.3)
        if bar_beat in [1, 3] and density > 0.3:
            events.append((beat, DRUM_SNARE, 0.4, 90))

        # Hi-hat on every beat (if density > 0.5)
        if density > 0.5:
            events.append((beat, DRUM_HAT, 0.2, int(55 + density * 35)))

        # Off-beat hi-hat (if density > 0.7)
        if density > 0.7 and bar_beat % 1 == 0:
            if beat + 0.5 < total:
                events.append((beat + 0.5, DRUM_HAT, 0.2, int(40 + density * 25)))

        # Random fills at end of bars (if density > 0.6)
        if density > 0.6 and bar_beat == bpb - 0.5 and random.random() < 0.4:
            events.append((beat, DRUM_SNARE, 0.2, 110))

        beat += 0.5   # Step in eighth notes

    return events


# ── Sound effects ──────────────────────────────────────────────────────────────

def _generate_sfx_shoot(output_path: str):
    """Short descending pitch — gunshot / laser effect."""
    midi = MIDIFile(1)
    midi.addTempo(0, 0, 180)
    # Quick descending notes
    for i, (note, beat, dur) in enumerate([
        (80, 0.0, 0.1),
        (72, 0.1, 0.1),
        (64, 0.2, 0.15),
    ]):
        midi.addNote(0, 0, note, beat, dur, 110 - i * 10)
    with open(output_path, "wb") as f:
        midi.writeFile(f)


def _generate_sfx_hit(output_path: str):
    """Short noise burst — impact / hit effect."""
    midi = MIDIFile(1)
    midi.addTempo(0, 0, 180)
    midi.addNote(0, 9, DRUM_SNARE, 0.0, 0.15, 127)
    midi.addNote(0, 9, DRUM_KICK,  0.1, 0.1,  100)
    with open(output_path, "wb") as f:
        midi.writeFile(f)


def _generate_sfx_wave_clear(output_path: str):
    """Ascending fanfare — wave complete jingle."""
    midi = MIDIFile(1)
    midi.addTempo(0, 0, 160)
    fanfare = [
        (60, 0.0,  0.2, 90),
        (64, 0.25, 0.2, 90),
        (67, 0.5,  0.2, 95),
        (72, 0.75, 0.5, 110),
    ]
    for note, beat, dur, vel in fanfare:
        midi.addNote(0, 0, note, beat, dur, vel)
    with open(output_path, "wb") as f:
        midi.writeFile(f)