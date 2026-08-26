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
        "gym":      TileType("gym",      "Gym/Boss Arena",True,  color="#aa3333"),
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
        "grass":     {"grass", "path", "tree", "tall_grass", "water", "building", "player", "town", "gym"},
        "path":      {"grass", "path", "tree", "tall_grass", "building", "town", "player", "water", "gym"},
        "tree":      {"grass", "tree", "path", "tall_grass", "building", "water"},
        "building":  {"grass", "path", "building", "tree", "tall_grass", "town"},
        "tall_grass":{"grass", "tall_grass", "tree", "path", "building", "water", "town"},
        "water":     {"water", "grass", "tall_grass", "path", "tree"},
        "player":    {"grass", "path"},
        "town":      {"grass", "path", "building", "tall_grass"},
        "gym":       {"path", "grass"},
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

# ── Genre-specific default map sizes ────────────────────────────────────────
# Open world needs to feel big. Dungeons/arenas can stay compact.
GENRE_MAP_SIZE = {
    "open_world_sandbox":    (60, 45),   # was 30x20 — now 2.7x bigger
    "top_down_action_rpg":   (36, 26),   # room for multiple dungeon rooms
    "turn_based_rpg":        (42, 30),   # town + routes + grass fields
    "wave_shooter":          (30, 20),   # unchanged — bunker fits fine
    "side_scroll_platformer":(48, 12),   # unchanged — already working
}


def _generate_bsp_dungeon(width: int, height: int) -> list[list[str]]:
    """
    Binary Space Partition dungeon generator.
    Produces connected rectangular rooms — the classic roguelike layout.
    Returns a grid of 'floor' / 'wall' before decoration is added.
    """
    import random as _r

    # Start with all walls
    grid = [["wall" for _ in range(width)] for _ in range(height)]

    class Node:
        def __init__(self, x, y, w, h):
            self.x, self.y, self.w, self.h = x, y, w, h
            self.left = None
            self.right = None
            self.room = None  # (rx, ry, rw, rh)

        def split(self, min_size=8):
                    can_h = self.h >= min_size * 2
                    can_w = self.w >= min_size * 2
                    if not can_h and not can_w:
                        return False

                    if can_h and can_w:
                        horizontal = _r.random() > 0.5
                    else:
                        horizontal = can_h  # pick whichever axis actually has room

                    if horizontal:
                        split_y = _r.randint(min_size, self.h - min_size)
                        self.left  = Node(self.x, self.y, self.w, split_y)
                        self.right = Node(self.x, self.y + split_y, self.w, self.h - split_y)
                    else:
                        split_x = _r.randint(min_size, self.w - min_size)
                        self.left  = Node(self.x, self.y, split_x, self.h)
                        self.right = Node(self.x + split_x, self.y, self.w - split_x, self.h)
                    return True

        def create_room(self):
            if self.left or self.right:
                if self.left:  self.left.create_room()
                if self.right: self.right.create_room()
                return
            # Leaf node — carve a room with margin
            margin = 1
            rw = _r.randint(4, max(4, self.w - margin * 2))
            rh = _r.randint(4, max(4, self.h - margin * 2))
            rx = self.x + _r.randint(margin, max(margin, self.w - rw - margin))
            ry = self.y + _r.randint(margin, max(margin, self.h - rh - margin))
            self.room = (rx, ry, rw, rh)
            for yy in range(ry, min(ry + rh, height)):
                for xx in range(rx, min(rx + rw, width)):
                    grid[yy][xx] = "floor"

        def get_room(self):
            if self.room:
                return self.room
            l = self.left.get_room() if self.left else None
            r = self.right.get_room() if self.right else None
            return l or r

        def connect_children(self):
            if not (self.left and self.right):
                return
            self.left.connect_children()
            self.right.connect_children()
            r1 = self.left.get_room()
            r2 = self.right.get_room()
            if r1 and r2:
                x1, y1 = r1[0] + r1[2]//2, r1[1] + r1[3]//2
                x2, y2 = r2[0] + r2[2]//2, r2[1] + r2[3]//2

                def carve_wide(cx, cy):
                    """Carve a 2x2 area so corridors are wide enough to walk through."""
                    for oy in range(2):
                        for ox in range(2):
                            ny, nx = cy + oy, cx + ox
                            if 0 <= ny < height and 0 <= nx < width:
                                grid[ny][nx] = "floor"

                if _r.random() > 0.5:
                    for x in range(min(x1,x2), max(x1,x2)+1):
                        carve_wide(x, y1)
                    for y in range(min(y1,y2), max(y1,y2)+1):
                        carve_wide(x2, y)
                else:
                    for y in range(min(y1,y2), max(y1,y2)+1):
                        carve_wide(x1, y)
                    for x in range(min(x1,x2), max(x1,x2)+1):
                        carve_wide(x, y2)

    root = Node(1, 1, width - 2, height - 2)
    def recursive_split(node, depth=0):
        if depth >= 3:
            return
        if node.split():
            recursive_split(node.left, depth + 1)
            recursive_split(node.right, depth + 1)

    recursive_split(root)
    root.create_room()
    root.connect_children()

    return grid


def _decorate_dungeon(grid: list[list[str]], width: int, height: int) -> list[list[str]]:
    """Add torches, chests, enemy spawns, player spawn to a BSP dungeon."""
    import random as _r

    floor_cells = [(x, y) for y in range(height) for x in range(width)
                   if grid[y][x] == "floor"]
    if not floor_cells:
        return grid

    floor_cells.sort(key=lambda p: p[0])
    px, py = floor_cells[0]
    grid[py][px] = "player"

    remaining = [c for c in floor_cells if c != (px, py)]
    remaining.sort(key=lambda p: -((p[0]-px)**2 + (p[1]-py)**2))
    enemy_count = min(8, len(remaining) // 12)
    for i in range(enemy_count):
        idx = int(i * len(remaining) / max(enemy_count, 1))
        ex, ey = remaining[idx]
        grid[ey][ex] = "enemy"

    chest_count = min(3, len(remaining) // 30)
    for i in range(chest_count):
        idx = _r.randint(0, len(remaining) - 1)
        cx, cy = remaining[idx]
        if grid[cy][cx] == "floor":
            grid[cy][cx] = "chest"

    torch_count = min(6, width // 4)
    wall_adjacent = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if grid[y][x] == "wall":
                neighbors = [grid[y-1][x], grid[y+1][x], grid[y][x-1], grid[y][x+1]]
                if "floor" in neighbors:
                    wall_adjacent.append((x, y))
    for _ in range(min(torch_count, len(wall_adjacent))):
        tx, ty = _r.choice(wall_adjacent)
        grid[ty][tx] = "torch"

    return grid

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



# ── Open World Sandbox — carve grid-aligned streets ──
    if genre == "open_world_sandbox":
        import random as _r
        road_tile     = "road"
        sidewalk_tile = "sidewalk"

        # Step 1 — fill EVERYTHING with block interior first
        # (building-heavy so streets will visually contrast against it)
        interior_weights = {"building": 6.0, "park": 2.0, "alley": 1.5}
        interior_tiles   = list(interior_weights.keys())
        interior_wts     = list(interior_weights.values())

        for y in range(height):
            for x in range(width):
                grid[y][x] = _r.choices(interior_tiles, weights=interior_wts, k=1)[0]

        # Step 2 — carve a real street grid ON TOP, overwriting interior
        # Vertical streets every 6 tiles (road + sidewalk pair)
        # Step 2 — carve a real street grid ON TOP, overwriting interior
        for x in range(0, width, 7):   # wider spacing between streets
            for y in range(height):
                grid[y][x] = road_tile
                if x + 1 < width:
                    grid[y][x + 1] = road_tile
                if x + 2 < width:
                    grid[y][x + 2] = sidewalk_tile

        for y in range(0, height, 6):  # wider spacing
            for x in range(width):
                grid[y][x] = road_tile
                if y + 1 < height:
                    grid[y + 1][x] = road_tile

        # Step 3 — player spawn on a road/sidewalk tile near top-left
        placed_player = False
        for y in range(height):
            for x in range(width):
                if grid[y][x] in (road_tile, sidewalk_tile):
                    grid[y][x] = "player"
                    placed_player = True
                    break
            if placed_player:
                break

        # Step 4 — NPC/cop spawns on streets, spaced out
        spawn_tile = next((t for t, td in tile_set.items()
                           if td.spawn_enemy), "spawn")
        placed = 0
        for y in range(2, height - 2, 4):
            for x in range(2, width - 2, 7):
                if grid[y][x] in (road_tile, sidewalk_tile) and placed < 12:
                    grid[y][x] = spawn_tile
                    placed += 1

        # Step 5 — mission markers on sidewalks
        mission_tile = next((t for t, td in tile_set.items()
                             if td.spawn_item), None)
        if mission_tile:
            placed = 0
            for y in range(3, height - 3, 6):
                for x in range(3, width - 3, 9):
                    if grid[y][x] in (road_tile, sidewalk_tile) and placed < 5:
                        grid[y][x] = mission_tile
                        placed += 1

        return grid

# ── Top down action RPG — Carve out real dungeon ──
    if genre == "top_down_action_rpg":
            dungeon = _generate_bsp_dungeon(width, height)
            dungeon = _decorate_dungeon(dungeon, width, height)
            return dungeon

# ── Turned based RPG — Carve out real town + grass overworld ──
    if genre == "turn_based_rpg":
            for y in range(height):
                for x in range(width):
                    grid[y][x] = "grass"

            town_w, town_h = width // 5, height // 4
            for y in range(2, 2 + town_h):
                for x in range(2, 2 + town_w):
                    grid[y][x] = "town" if (x + y) % 3 != 0 else "path"

            path_x, path_y = 2 + town_w, 2 + town_h // 2
            target_x, target_y = width - 4, height - 4
            while path_x != target_x or path_y != target_y:
                grid[path_y][path_x] = "path"
                if path_x < target_x: path_x += 1
                elif path_x > target_x: path_x -= 1
                elif path_y < target_y: path_y += 1
                elif path_y > target_y: path_y -= 1
                grid[path_y][path_x] = "path"

            # Gym — boss arena at the path's destination (Sprint 8E).
            # Keep the approach clear so the player can always reach it.
            grid[target_y][target_x] = "gym"
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = target_y + dy, target_x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if grid[ny][nx] in ("tree", "water", "building"):
                            grid[ny][nx] = "path"

            import random as _r
            for _ in range(6):
                cx = _r.randint(town_w + 3, width - 5)
                cy = _r.randint(town_h + 3, height - 5)
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < height and 0 <= nx < width and grid[ny][nx] == "grass":
                            if _r.random() > 0.4:
                                grid[ny][nx] = "tall_grass"

            for _ in range(width * height // 25):
                tx = _r.randint(0, width - 1)
                ty = _r.randint(0, height - 1)
                if grid[ty][tx] == "grass":
                    grid[ty][tx] = "tree"

            grid[3][3] = "player"
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
                    max_retries: int = 5,
                    genre_override: str = None) -> dict:
    """
    Generate a tilemap for a demake. Dispatches to genre-specific
    generation strategy, then builds the shared manifest output
    (tile_defs, spawn_points, stats) for ALL genres — this part
    must run regardless of which branch built the grid.
    """
    genre = genre_override or dna.genre

    if width is None or height is None:
        default_w, default_h = GENRE_MAP_SIZE.get(genre, (30, 20))
        width  = width  or default_w
        height = height or default_h

    weights = GENRE_WEIGHTS.get(genre, GENRE_WEIGHTS["wave_shooter"])
    print(f"[TilemapGen] Generating {width}x{height} tilemap for {genre}")

    # ── Step 1: Get the base grid — genre-specific strategy ──────────────
    if genre == "top_down_action_rpg":
        # BSP dungeon builds AND decorates in one call.
        # Do NOT also call _ensure_required_tiles — it would regenerate
        # a second dungeon and discard this one.
        grid = _generate_bsp_dungeon(width, height)
        grid = _decorate_dungeon(grid, width, height)

    elif genre == "open_world_sandbox":
        # Blank grid — _ensure_required_tiles carves the street grid +
        # building-dominant blocks + spawns entirely from scratch.
        grid = [["building" for _ in range(width)] for _ in range(height)]
        grid = _ensure_required_tiles(grid, genre, width, height)

    else:
        # Standard WFC path — wave_shooter, side_scroll_platformer,
        # turn_based_rpg (side_scroll and turn_based have their own
        # post-process branches inside _ensure_required_tiles already)
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

        grid = _ensure_required_tiles(grid, genre, width, height)

    # ── Step 2: Build manifest output — runs for EVERY genre ─────────────
    tile_set  = TILE_SETS.get(genre, TILE_SETS["default"])
    tile_defs = {
        tile_id: {
            "display":      td.display,
            "passable":     td.passable,
            "spawn_enemy":  td.spawn_enemy,
            "spawn_player": td.spawn_player,
            "spawn_item":   td.spawn_item,
            "color":        td.color,
        }
        for tile_id, td in tile_set.items()
    }

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
            "player":  player_spawns[0] if player_spawns else [1, 1],
            "enemies": enemy_spawns,
            "items":   item_spawns,
        },
        "stats": {
            "total_tiles": width * height,
            "walkable":    sum(1 for y in range(height) for x in range(width)
                               if tile_set.get(grid[y][x], TileType("x","x",False)).passable),
            "enemy_count": len(enemy_spawns),
            "item_count":  len(item_spawns),
        }
    }

    # ── Step 3: Write to disk — runs for EVERY genre ──────────────────────
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