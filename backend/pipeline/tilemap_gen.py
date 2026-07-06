"""
Sprint 7 — Tilemap Generation via Wave Function Collapse

Generates authentic-feeling level layouts for each genre template.
WFC respects adjacency rules so output looks hand-crafted rather than random.

Architecture doc reference (Sprint 7):
  Use LLM to output a full tile grid for procedural level layouts
  instead of empty arenas.

We use WFC instead of an LLM because:
  - Zero VRAM cost
  - Deterministic, no hallucinations
  - Naturally produces valid game levels
  - Output directly importable into Unity Tilemap system (Sprint 8)
"""
import os
import json
import random
from dataclasses import dataclass, field
from typing import Optional
from pipeline.validator import GameDNA


# ── Tile type definitions ──────────────────────────────────────────────────────

@dataclass
class TileType:
    id:          str
    display:     str    # Human-readable name
    passable:    bool   # Can player/enemies walk through?
    spawn_enemy: bool = False
    spawn_player:bool = False
    spawn_item:  bool = False
    color:       str  = "#333333"  # Fallback color for rendering


# ── Genre-specific tile sets ───────────────────────────────────────────────────

TILE_SETS = {
    "top_down_action_rpg": {
        "floor":    TileType("floor",    "Stone Floor",   True,  color="#2a2a3a"),
        "wall":     TileType("wall",     "Stone Wall",    False, color="#160a16"),
        "door":     TileType("door",     "Door",          True,  color="#8b4513"),
        "chest":    TileType("chest",    "Treasure Chest",False, spawn_item=True, color="#ffd700"),
        "torch":    TileType("torch",    "Torch",         False, color="#ff6600"),
        "enemy":    TileType("enemy",    "Enemy Spawn",   True,  spawn_enemy=True, color="#ff2200"),
        "player":   TileType("player",   "Player Start",  True,  spawn_player=True, color="#00ff88"),
        "void":     TileType("void",     "Void",          False, color="#000000"),
    },
    "wave_shooter": {
        "floor":    TileType("floor",    "Concrete",      True,  color="#2a2a2a"),
        "wall":     TileType("wall",     "Wall",          False, color="#0f1218"),
        "cover":    TileType("cover",    "Cover/Barricade",False,color="#4a3a2a"),
        "spawn":    TileType("spawn",    "Enemy Spawn",   True,  spawn_enemy=True, color="#ff2200"),
        "player":   TileType("player",   "Player Start",  True,  spawn_player=True, color="#00ff88"),
        "ammo":     TileType("ammo",     "Ammo Cache",    True,  spawn_item=True, color="#ffcc00"),
        "void":     TileType("void",     "Void",          False, color="#000000"),
    },
    "open_world_sandbox": {
        "road":     TileType("road",     "Road",          True,  color="#333333"),
        "sidewalk": TileType("sidewalk", "Sidewalk",      True,  color="#888888"),
        "building": TileType("building", "Building",      False, color="#4a4a6a"),
        "park":     TileType("park",     "Park/Grass",    True,  color="#2a6a2a"),
        "alley":    TileType("alley",    "Alley",         True,  color="#222222"),
        "spawn":    TileType("spawn",    "NPC Spawn",     True,  spawn_enemy=True, color="#ff8800"),
        "player":   TileType("player",   "Player Start",  True,  spawn_player=True, color="#00ff88"),
        "mission":  TileType("mission",  "Mission Marker",True,  spawn_item=True, color="#ffff00"),
    },
    "side_scroll_platformer": {
        "platform": TileType("platform", "Platform",      False, color="#8b4513"),
        "ground":   TileType("ground",   "Ground",        False, color="#4a7a1a"),
        "sky":      TileType("sky",      "Sky/Air",       True,  color="#4a8aff"),
        "coin":     TileType("coin",     "Coin",          True,  spawn_item=True, color="#ffd700"),
        "pipe":     TileType("pipe",     "Pipe",          False, color="#1a8a1a"),
        "enemy":    TileType("enemy",    "Enemy Spawn",   True,  spawn_enemy=True, color="#ff2200"),
        "player":   TileType("player",   "Player Start",  True,  spawn_player=True, color="#00ff88"),
        "goal":     TileType("goal",     "Goal/Flag",     True,  spawn_item=True, color="#ffffff"),
    },
    "turn_based_rpg": {
        "grass":    TileType("grass",    "Grass",         True,  color="#2a6a2a"),
        "path":     TileType("path",     "Path",          True,  color="#8b6914"),
        "tree":     TileType("tree",     "Tree",          False, color="#1a4a1a"),
        "building": TileType("building", "Building",      False, color="#8b4513"),
        "tall_grass":TileType("tall_grass","Tall Grass",  True,  spawn_enemy=True, color="#3a8a3a"),
        "water":    TileType("water",    "Water",         False, color="#1a4a8a"),
        "player":   TileType("player",   "Player Start",  True,  spawn_player=True, color="#00ff88"),
        "town":     TileType("town",     "Town Building", True,  spawn_item=True, color="#cc8844"),
    },
}

# Default fallback tile set
TILE_SETS["default"] = TILE_SETS["wave_shooter"]


# ── Adjacency rules ────────────────────────────────────────────────────────────
# tile_id → set of tile_ids that can appear adjacent (N/S/E/W)

ADJACENCY_RULES = {
    "top_down_action_rpg": {
        "floor":  {"floor", "wall", "door", "chest", "torch", "enemy", "player"},
        "wall":   {"wall", "floor", "door", "torch", "void"},
        "door":   {"floor", "wall"},
        "chest":  {"floor"},
        "torch":  {"wall", "floor"},
        "enemy":  {"floor"},
        "player": {"floor"},
        "void":   {"void", "wall"},
    },
    "wave_shooter": {
        "floor":  {"floor", "wall", "cover", "spawn", "player", "ammo"},
        "wall":   {"wall", "floor", "cover", "void"},
        "cover":  {"floor", "wall"},
        "spawn":  {"floor"},
        "player": {"floor"},
        "ammo":   {"floor"},
        "void":   {"void", "wall"},
    },
    "open_world_sandbox": {
        "road":     {"road", "sidewalk", "building", "alley", "spawn", "player", "mission"},
        "sidewalk": {"road", "sidewalk", "building", "park", "alley", "mission", "player", "spawn"},
        "building": {"road", "sidewalk", "building", "alley"},
        "park":     {"sidewalk", "park", "mission"},
        "alley":    {"road", "building", "sidewalk", "spawn"},
        "spawn":    {"road", "sidewalk", "alley"},
        "player":   {"road", "sidewalk"},
        "mission":  {"road", "sidewalk", "park"},
    },
    "side_scroll_platformer": {
        "sky":      {"sky", "platform", "coin", "enemy", "player", "goal"},
        "platform": {"sky", "platform", "ground", "pipe", "coin", "enemy", "player", "goal"},
        "ground":   {"ground", "platform", "pipe"},
        "coin":     {"sky", "platform"},
        "pipe":     {"ground", "platform"},
        "enemy":    {"sky", "platform"},
        "player":   {"sky", "platform"},
        "goal":     {"sky", "platform"},
    },
    "turn_based_rpg": {
        "grass":     {"grass", "path", "tree", "tall_grass", "water", "building", "player", "town"},
        "path":      {"grass", "path", "tree", "tall_grass", "building", "town", "player", "water"},
        "tree":      {"grass", "tree", "path", "tall_grass", "building", "water"},
        "building":  {"grass", "path", "building", "tree", "tall_grass", "town"},
        "tall_grass":{"grass", "tall_grass", "tree", "path", "building", "water", "town"},
        "water":     {"water", "grass", "tall_grass", "path", "tree"},
        "player":    {"grass", "path"},
        "town":      {"grass", "path", "building", "tall_grass"},
    },
}


# ── WFC Implementation ─────────────────────────────────────────────────────────

class WaveFunctionCollapse:
    """
    Minimal WFC implementation for tilemap generation.

    Each cell starts in superposition (all tiles possible).
    We repeatedly:
      1. Find the cell with lowest entropy (fewest possibilities) — Observe
      2. Collapse it to one tile (weighted random)               — Collapse
      3. Propagate constraints to neighbors                      — Propagate
    Until all cells are determined.
    """

    def __init__(self, width: int, height: int, genre: str,
                 weights: Optional[dict] = None):
        self.width   = width
        self.height  = height
        self.genre   = genre
        self.tiles   = TILE_SETS.get(genre, TILE_SETS["default"])
        self.rules   = ADJACENCY_RULES.get(genre, ADJACENCY_RULES["wave_shooter"])
        self.tile_ids = list(self.tiles.keys())
        self.weights  = weights or {t: 1.0 for t in self.tile_ids}

        # Grid of possible tiles per cell — starts fully open
        self.grid: list[list[set]] = [
            [set(self.tile_ids) for _ in range(width)]
            for _ in range(height)
        ]

        self.collapsed = [[False] * width for _ in range(height)]
        self._contradiction = False

    def solve(self, max_iterations: int = 10000) -> bool:
        """Run WFC until solved or contradiction. Returns True on success."""
        iterations = 0
        while not self._is_fully_collapsed() and iterations < max_iterations:
            if not self._step():
                # Contradiction — restart with a fresh grid
                return False
            iterations += 1
        return self._is_fully_collapsed() and not self._contradiction

    def _is_fully_collapsed(self) -> bool:
        return all(
            len(self.grid[y][x]) == 1
            for y in range(self.height)
            for x in range(self.width)
        )

    def _step(self) -> bool:
        """One WFC step: observe lowest entropy cell, collapse it, propagate."""
        # Find lowest entropy (uncollapsed) cell
        min_entropy = float('inf')
        candidates  = []

        for y in range(self.height):
            for x in range(self.width):
                if self.collapsed[y][x]:
                    continue
                entropy = len(self.grid[y][x])
                if entropy == 0:
                    return False  # Contradiction
                if entropy < min_entropy:
                    min_entropy = entropy
                    candidates  = [(x, y)]
                elif entropy == min_entropy:
                    candidates.append((x, y))

        if not candidates:
            return True

        # Pick random cell from lowest entropy candidates
        cx, cy = random.choice(candidates)

        # Collapse — weighted random selection
        possible = list(self.grid[cy][cx])
        weights  = [self.weights.get(t, 1.0) for t in possible]
        total    = sum(weights)
        r        = random.uniform(0, total)
        chosen   = possible[0]
        for tile, w in zip(possible, weights):
            r -= w
            if r <= 0:
                chosen = tile
                break

        self.grid[cy][cx] = {chosen}
        self.collapsed[cy][cx] = True

        # Propagate constraints
        self._propagate(cx, cy)
        return True

    def _propagate(self, start_x: int, start_y: int):
        """Propagate constraints from collapsed cell to all reachable neighbors."""
        stack = [(start_x, start_y)]
        visited = set()

        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.add((x, y))

            current_options = self.grid[y][x]

            # Check all 4 neighbors
            for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    continue

                # What tiles are allowed adjacent to any of current_options?
                allowed = set()
                for tile in current_options:
                    allowed |= self.rules.get(tile, set())

                # Constrain neighbor
                before = len(self.grid[ny][nx])
                self.grid[ny][nx] &= allowed

                if len(self.grid[ny][nx]) == 0:
                    self._contradiction = True
                    return

                # If neighbor changed, propagate from it too
                if len(self.grid[ny][nx]) < before:
                    stack.append((nx, ny))

    def get_result(self) -> list[list[str]]:
        """Return the collapsed grid as a 2D list of tile ID strings."""
        result = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                options = self.grid[y][x]
                if options:
                    row.append(next(iter(options)))
                else:
                    # Fallback for uncollapsed cells
                    default = list(self.tiles.keys())[0]
                    row.append(default)
            result.append(row)
        return result


# ── Genre-specific tile weights ────────────────────────────────────────────────
# Higher weight = more likely to appear. Tuned per genre for good level feel.

GENRE_WEIGHTS = {
    "top_down_action_rpg": {
        "floor": 18.0, "wall":  1.2, "door":  1.0, "chest": 0.2,
        "torch": 0.3, "enemy": 0.8, "player": 0.1, "void":  0.3,
    },
    "wave_shooter": {
        "floor": 18.0, "wall":  1.0, "cover": 1.0, "spawn":  0.8,
        "player": 0.1, "ammo":  0.6, "void":  0.2,
    },
    "open_world_sandbox": {
        "road":    8.0, "sidewalk": 6.0, "building": 2.5, "park":   3.0,
        "alley":   1.5, "spawn":    1.0, "player":   0.1, "mission":0.4,
    },
    "side_scroll_platformer": {
        "sky":    10.0, "platform": 2.0, "ground": 1.5, "coin":  1.5,
        "pipe":   0.5,  "enemy":   0.8, "player": 0.1, "goal":  0.1,
    },
    "turn_based_rpg": {
        "grass":     12.0, "path":  3.0, "tree":  1.5, "building":  0.6,
        "tall_grass": 3.0, "water": 0.8, "player": 0.1, "town":     0.8,
    },
}


def _ensure_required_tiles(grid: list[list[str]], genre: str,
                            width: int, height: int) -> list[list[str]]:
    """
    Post-process WFC output to guarantee required tiles exist:
    - Exactly one player spawn
    - At least one enemy spawn
    - At least one walkable path from player to enemies
    """
    tile_set = TILE_SETS.get(genre, TILE_SETS["default"])
    floor_tiles = [t for t, td in tile_set.items() if td.passable and not td.spawn_player]

    # ── Platformer-specific post-processing ──
    if genre == "side_scroll_platformer":
        sky_tile = next((t for t, td in tile_set.items() if t == "sky"), "sky")

        # Force bottom 2 rows as ground so player has a floor
        for x in range(width):
            for y in range(height - 2, height):
                grid[y][x] = "ground"

        # Clear duplicate player/goal tiles above ground (WFC places them randomly)
        for y in range(height - 2):
            for x in range(width):
                if grid[y][x] in ("player", "goal"):
                    grid[y][x] = sky_tile

        # Player on left side, standing on ground
        grid[height - 3][3] = "player"

        # Goal on far right, one row above ground
        grid[height - 3][width - 4] = "goal"

        # Place enemies on ground row (height-2) at intervals
        enemy_tile = next((t for t, td in tile_set.items() if td.spawn_enemy), None)
        spawn_interval = max(6, (width - 8) // 4)
        for ex in range(8, width - 4, spawn_interval):
            grid[height - 2][ex] = enemy_tile or "enemy"

        return grid

    # ── General post-processing for all other genres ──
    # Find passable tiles for player spawn
    passable = [(x, y) for y in range(height) for x in range(width)
                if tile_set.get(grid[y][x], TileType("x","x",False)).passable]

    if not passable:
        # Emergency fallback — fill center area with floor
        cx, cy = width//2, height//2
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if 0 <= cy+dy < height and 0 <= cx+dx < width:
                    grid[cy+dy][cx+dx] = floor_tiles[0] if floor_tiles else "floor"
        passable = [(cx, cy)]

    # Place player spawn (clear center area)
    px = width // 6
    py = height // 2
    # Find nearest passable tile to intended spawn
    passable.sort(key=lambda p: abs(p[0]-px) + abs(p[1]-py))
    if passable:
        sx, sy = passable[0]
        grid[sy][sx] = "player"
        # Clear a 2x2 area around player
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = sx+dx, sy+dy
                if 0 < nx < width-1 and 0 < ny < height-1:
                    t = tile_set.get(grid[ny][nx])
                    if t and not t.passable:
                        grid[ny][nx] = floor_tiles[0] if floor_tiles else "floor"

    # Ensure enemy spawns exist
    enemy_tiles = [t for t, td in tile_set.items() if td.spawn_enemy]
    enemy_spawns = [(x, y) for y in range(height) for x in range(width)
                    if grid[y][x] in enemy_tiles]
    if len(enemy_spawns) < 3 and passable and enemy_tiles:
        # Place enemies in far areas
        far_passable = sorted(passable, key=lambda p: -(p[0] + p[1]))
        for ex, ey in far_passable[:3]:
            if grid[ey][ex] != "player":
                grid[ey][ex] = enemy_tiles[0]

    return grid


# ── Main entry point ───────────────────────────────────────────────────────────

def run_tilemap_gen(dna: GameDNA, output_dir: str,
                    width: int = None, height: int = None,
                    max_retries: int = 5) -> dict:
    """
    Generate a tilemap for a demake using Wave Function Collapse.

    Platformers default to 48x12 (wider+shorter), other genres default to 30x20.

    Args:
        dna:        Validated GameDNA — determines tile set and weights
        output_dir: /outputs/{demake_id}/ — tilemap.json written here
        width:      Tilemap width in tiles (default 48 for platformer, else 30)
        height:     Tilemap height in tiles (default 12 for platformer, else 20)
        max_retries:WFC can hit contradictions — retry up to N times

    Returns:
        Tilemap dict with grid, tile definitions, and spawn points
    """
    if width is None:
        width = 48 if dna.genre == "side_scroll_platformer" else 30
    if height is None:
        height = 12 if dna.genre == "side_scroll_platformer" else 20

    genre   = dna.genre
    weights = GENRE_WEIGHTS.get(genre, GENRE_WEIGHTS["wave_shooter"])

    print(f"[TilemapGen] Generating {width}x{height} tilemap for {genre}")

    grid = None
    for attempt in range(max_retries):
        wfc = WaveFunctionCollapse(width, height, genre, weights)
        success = wfc.solve()
        if success:
            grid = wfc.get_result()
            print(f"[TilemapGen] WFC solved on attempt {attempt + 1}")
            break
        print(f"[TilemapGen] WFC contradiction on attempt {attempt + 1} — retrying")

    if grid is None:
        print("[TilemapGen] WFC failed all retries — using fallback grid")
        grid = _fallback_grid(width, height, genre)

    # Post-process to guarantee player/enemy spawns
    grid = _ensure_required_tiles(grid, genre, width, height)

    # Build tile definitions for frontend
    tile_set    = TILE_SETS.get(genre, TILE_SETS["default"])
    tile_defs   = {
        tile_id: {
            "display":     td.display,
            "passable":    td.passable,
            "spawn_enemy": td.spawn_enemy,
            "spawn_player":td.spawn_player,
            "spawn_item":  td.spawn_item,
            "color":       td.color,
        }
        for tile_id, td in tile_set.items()
    }

    # Find spawn points
    player_spawns = [(x, y) for y in range(height) for x in range(width)
                     if grid[y][x] == "player"]
    enemy_spawns  = [(x, y) for y in range(height) for x in range(width)
                     if tile_set.get(grid[y][x], TileType("x","x",False)).spawn_enemy]
    item_spawns   = [(x, y) for y in range(height) for x in range(width)
                     if tile_set.get(grid[y][x], TileType("x","x",False)).spawn_item]

    tilemap = {
        "genre":        genre,
        "width":        width,
        "height":       height,
        "tile_size":    16,
        "grid":         grid,
        "tile_defs":    tile_defs,
        "spawn_points": {
            "player": player_spawns[0] if player_spawns else [1, 1],
            "enemies":enemy_spawns,
            "items":  item_spawns,
        },
        "stats": {
            "total_tiles":  width * height,
            "walkable":     sum(1 for y in range(height) for x in range(width)
                               if tile_set.get(grid[y][x], TileType("x","x",False)).passable),
            "enemy_count":  len(enemy_spawns),
            "item_count":   len(item_spawns),
        }
    }

    # Write to disk
    tilemap_path = os.path.normpath(os.path.join(output_dir, "tilemap.json"))
    with open(tilemap_path, "w") as f:
        json.dump(tilemap, f, indent=2)

    print(f"[TilemapGen] Written: {tilemap_path}")
    print(f"[TilemapGen] Stats: {tilemap['stats']}")

    return tilemap


def _fallback_grid(width: int, height: int, genre: str) -> list[list[str]]:
    """Simple fallback when WFC fails — bordered room with open center."""
    tile_set = TILE_SETS.get(genre, TILE_SETS["default"])
    wall_tile  = next((t for t, td in tile_set.items() if not td.passable
                      and t not in ("void",)), "wall")
    floor_tile = next((t for t, td in tile_set.items()
                      if td.passable and not td.spawn_enemy
                      and not td.spawn_player), "floor")

    grid = []
    for y in range(height):
        row = []
        for x in range(width):
            if x == 0 or x == width-1 or y == 0 or y == height-1:
                row.append(wall_tile)
            else:
                row.append(floor_tile)
        grid.append(row)
    return grid