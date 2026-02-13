"""Renderer - converts game state to 64x64 RGB frames."""

from dataclasses import dataclass
from typing import Optional
from .game_object import GameObject, Position


@dataclass
class Renderer:
    """Renders a game level to a 64x64 RGB grid."""
    
    width: int = 64
    height: int = 64
    background_color: tuple[int, int, int] = (0, 0, 0)
    
    def create_empty_frame(self) -> list[list[list[int]]]:
        return [[list(self.background_color) for _ in range(self.width)] for _ in range(self.height)]
    
    def render_objects(self, objects: list[GameObject], frame: Optional[list[list[list[int]]]] = None) -> list[list[list[int]]]:
        if frame is None:
            frame = self.create_empty_frame()
        for obj in objects:
            if not obj.visible:
                continue
            for pos, color in obj.render():
                if 0 <= pos.x < self.width and 0 <= pos.y < self.height:
                    frame[pos.y][pos.x] = list(color)
        return frame
    
    def render_grid(self, grid: list[list[int]], color_map: dict[int, tuple[int, int, int]], 
                    frame: Optional[list[list[list[int]]]] = None) -> list[list[list[int]]]:
        if frame is None:
            frame = self.create_empty_frame()
        for y, row in enumerate(grid):
            if y >= self.height:
                break
            for x, tile in enumerate(row):
                if x >= self.width:
                    break
                if tile in color_map:
                    frame[y][x] = list(color_map[tile])
        return frame
    
    def draw_rect(self, frame: list[list[list[int]]], x: int, y: int, width: int, height: int,
                  color: tuple[int, int, int], filled: bool = True) -> list[list[list[int]]]:
        for dy in range(height):
            for dx in range(width):
                px, py = x + dx, y + dy
                if 0 <= px < self.width and 0 <= py < self.height:
                    if filled or dx == 0 or dx == width - 1 or dy == 0 or dy == height - 1:
                        frame[py][px] = list(color)
        return frame
    
    def draw_line(self, frame: list[list[list[int]]], x1: int, y1: int, x2: int, y2: int,
                  color: tuple[int, int, int]) -> list[list[list[int]]]:
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        x, y = x1, y1
        while True:
            if 0 <= x < self.width and 0 <= y < self.height:
                frame[y][x] = list(color)
            if x == x2 and y == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return frame
    
    def draw_text_simple(self, frame: list[list[list[int]]], x: int, y: int, char: str,
                         color: tuple[int, int, int], scale: int = 1) -> list[list[list[int]]]:
        FONT = {
            '0': ['111', '101', '101', '101', '111'],
            '1': ['010', '110', '010', '010', '111'],
            '2': ['111', '001', '111', '100', '111'],
            '3': ['111', '001', '111', '001', '111'],
            '4': ['101', '101', '111', '001', '001'],
            '5': ['111', '100', '111', '001', '111'],
            '6': ['111', '100', '111', '101', '111'],
            '7': ['111', '001', '001', '001', '001'],
            '8': ['111', '101', '111', '101', '111'],
            '9': ['111', '101', '111', '001', '111'],
        }
        if char not in FONT:
            return frame
        pattern = FONT[char]
        for row_idx, row in enumerate(pattern):
            for col_idx, pixel in enumerate(row):
                if pixel == '1':
                    for sy in range(scale):
                        for sx in range(scale):
                            px = x + col_idx * scale + sx
                            py = y + row_idx * scale + sy
                            if 0 <= px < self.width and 0 <= py < self.height:
                                frame[py][px] = list(color)
        return frame
