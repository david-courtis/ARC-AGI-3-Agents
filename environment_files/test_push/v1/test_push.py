"""
Test Push Game — a minimal Sokoban-like puzzle for pipeline validation.

KNOWN RULES (ground truth for evaluating the agent):
  - Player (color 1, blue 3x3): moves with ACTION1=up, ACTION2=down, ACTION3=left, ACTION4=right
  - Wall (color 11, grey 3x3): static, blocks player and box movement
  - Box (color 8, orange 3x3): pushed by player. When player moves into box,
    box slides 1 step in that direction IF the cell behind is empty.
  - Goal (color 3, green 3x3): marks the target cell. Box landing on goal = level complete.
  - ACTION5: no effect (no-op)
  - ACTION6: not available

Level 1: Simple — push box right onto goal
  Layout (in grid cells of 3px each, 0-indexed):
    Row 0: all wall (border)
    Row 1: wall, empty, Player(1,1), empty, Box(1,4), empty, Goal(1,6), wall
    Row 2: all wall (border)

  The agent must: move right 3 times (pushing box each time) to land box on goal.

Level 2: Slightly harder — push box around a wall
  Player starts left, box in middle, wall blocks direct path, goal is right.
"""

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite


# Sprites — all 3x3 blocks for clean detection
_3x3 = lambda color: [[color] * 3 for _ in range(3)]

sprites = {
    "player": Sprite(
        pixels=_3x3(1),  # color 1 = blue
        name="player",
        collidable=True,
        tags=["player"],
    ),
    "wall": Sprite(
        pixels=_3x3(11),  # color 11 = grey
        name="wall",
        collidable=True,
        tags=["wall"],
    ),
    "box": Sprite(
        pixels=_3x3(8),  # color 8 = orange
        name="box",
        collidable=True,
        tags=["box"],
    ),
    "goal": Sprite(
        pixels=_3x3(3),  # color 3 = green
        name="goal",
        collidable=False,  # player and box can walk over it
        tags=["goal"],
    ),
}


def _wall(col, row):
    """Create a wall sprite at grid position (col, row). Each cell is 3px."""
    return sprites["wall"].clone().set_position(col * 3, row * 3)


# Level 1: simple push right
# 8 columns x 3 rows, each cell 3px → 24x9 pixels (camera will be 64x64, centered)
_level1_sprites = []

# Top and bottom wall rows
for c in range(8):
    _level1_sprites.append(_wall(c, 0))
    _level1_sprites.append(_wall(c, 2))
# Left and right walls
_level1_sprites.append(_wall(0, 1))
_level1_sprites.append(_wall(7, 1))

# Player at (1, 1) in grid coords
_level1_sprites.append(sprites["player"].clone().set_position(1 * 3, 1 * 3))
# Box at (4, 1)
_level1_sprites.append(sprites["box"].clone().set_position(4 * 3, 1 * 3))
# Goal at (6, 1)
_level1_sprites.append(sprites["goal"].clone().set_position(6 * 3, 1 * 3))

# Level 2: push around corner
# 8 columns x 5 rows
_level2_sprites = []

# Border walls
for c in range(8):
    _level2_sprites.append(_wall(c, 0))
    _level2_sprites.append(_wall(c, 4))
for r in range(1, 4):
    _level2_sprites.append(_wall(0, r))
    _level2_sprites.append(_wall(7, r))

# Internal wall blocking direct path
_level2_sprites.append(_wall(4, 1))
_level2_sprites.append(_wall(4, 2))

# Player at (1, 2)
_level2_sprites.append(sprites["player"].clone().set_position(1 * 3, 2 * 3))
# Box at (3, 2)
_level2_sprites.append(sprites["box"].clone().set_position(3 * 3, 2 * 3))
# Goal at (6, 2)
_level2_sprites.append(sprites["goal"].clone().set_position(6 * 3, 2 * 3))


levels = [
    Level(
        sprites=_level1_sprites,
        grid_size=(64, 64),
        name="Level 1 - Push Right",
    ),
    Level(
        sprites=_level2_sprites,
        grid_size=(64, 64),
        name="Level 2 - Push Around Wall",
    ),
]


class TestPush(ARCBaseGame):
    def __init__(self) -> None:
        super().__init__(
            game_id="test_push",
            levels=levels,
            camera=Camera(width=64, height=64, background=0),
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5],
        )

    def on_set_level(self, level: Level) -> None:
        pass

    def step(self) -> None:
        action = self.action.id

        player_list = self.current_level.get_sprites_by_tag("player")
        if not player_list:
            self.complete_action()
            return
        player = player_list[0]

        # Directional movement
        dx, dy = 0, 0
        if action == GameAction.ACTION1:
            dy = -3  # up (3px = 1 grid cell)
        elif action == GameAction.ACTION2:
            dy = 3   # down
        elif action == GameAction.ACTION3:
            dx = -3  # left
        elif action == GameAction.ACTION4:
            dx = 3   # right
        elif action == GameAction.ACTION5:
            # No-op
            self.complete_action()
            return
        else:
            self.complete_action()
            return

        # Check what's at the target position
        target_x = player.x + dx
        target_y = player.y + dy

        # Check for box at target
        box_list = self.current_level.get_sprites_by_tag("box")
        box_at_target = None
        for box in box_list:
            if box.x == target_x and box.y == target_y:
                box_at_target = box
                break

        if box_at_target:
            # Try to push the box
            box_dest_x = box_at_target.x + dx
            box_dest_y = box_at_target.y + dy

            # Check if box destination is free (no wall, no other box)
            can_push = True

            # Check walls
            for wall in self.current_level.get_sprites_by_tag("wall"):
                if wall.x == box_dest_x and wall.y == box_dest_y:
                    can_push = False
                    break

            # Check other boxes
            if can_push:
                for other_box in box_list:
                    if other_box is not box_at_target:
                        if other_box.x == box_dest_x and other_box.y == box_dest_y:
                            can_push = False
                            break

            if can_push:
                # Push: move box, then move player
                box_at_target.set_position(box_dest_x, box_dest_y)
                player.set_position(target_x, target_y)

                # Check win: box on goal
                for goal in self.current_level.get_sprites_by_tag("goal"):
                    if box_at_target.x == goal.x and box_at_target.y == goal.y:
                        if self.is_last_level():
                            self.win()
                        else:
                            self.next_level()
                        self.complete_action()
                        return
            # else: can't push, player doesn't move

        else:
            # No box — check for wall collision
            blocked = False
            for wall in self.current_level.get_sprites_by_tag("wall"):
                if wall.x == target_x and wall.y == target_y:
                    blocked = True
                    break

            if not blocked:
                player.set_position(target_x, target_y)

        self.complete_action()
