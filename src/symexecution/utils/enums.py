from enum import Enum


class Direction(str, Enum):
    north = 'north'
    east = 'east'
    south = 'south'
    west = 'west'
    any = 'any'

    def __str__(self):
        return self.value


class Quadrant(str, Enum):
    top_left = 'top_left'
    top_right = 'top_right'
    bottom_left = 'bottom_left'
    bottom_right = 'bottom_right'
    center = 'center'

    def __str__(self):
        return self.value


idx2quad_dir = {
    0: (Quadrant.center, Direction.north),
    1: (Quadrant.bottom_left, Direction.north),
    2: (Quadrant.top_left, Direction.north),
    3: (Quadrant.bottom_right, Direction.north),
    4: (Quadrant.top_right, Direction.north),
    5: (Quadrant.center, Direction.east),
    6: (Quadrant.bottom_left, Direction.east),
    7: (Quadrant.top_left, Direction.east),
    8: (Quadrant.bottom_right, Direction.east),
    9: (Quadrant.top_right, Direction.east),
    10: (Quadrant.center, Direction.south),
    11: (Quadrant.bottom_left, Direction.south),
    12: (Quadrant.top_left, Direction.south),
    13: (Quadrant.bottom_right, Direction.south),
    14: (Quadrant.top_right, Direction.south),
    15: (Quadrant.center, Direction.west),
    16: (Quadrant.bottom_left, Direction.west),
    17: (Quadrant.top_left, Direction.west),
    18: (Quadrant.bottom_right, Direction.west),
    19: (Quadrant.top_right, Direction.west),
}

quad_dir2idx = {v: k for k, v in idx2quad_dir.items()}
