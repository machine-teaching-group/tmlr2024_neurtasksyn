from typing import Optional

import numpy as np
import torch

from src.symexecution.decision_makers import DecisionMaker, RandomDecisionMaker
from src.symexecution.utils.enums import Direction, Quadrant, idx2quad_dir
from src.symexecution.utils.quadrants import get_position_from_quadrant
from src.utils.colors import Tcolors

MATRIX_DIMENSIONS = 4
MAX_MARKERS_PER_SQUARE = 100

class SymWorld:

    def __init__(self, rows: int, cols: int,
                 hero_row: Optional[int], hero_col: Optional[int],
                 hero_dir: Optional[Direction],
                 blocked: Optional[np.ndarray],
                 markers: Optional[np.ndarray],
                 unknown: Optional[np.ndarray],
                 decision_maker: DecisionMaker = None,
                 original_markers: Optional[np.ndarray] = None):
        self.rows = rows
        self.cols = cols

        self.heroRow = hero_row
        self.heroCol = hero_col
        self.heroDir = hero_dir

        self.blocked = blocked
        self.markers = markers

        self.crashed = False

        self.unknown = unknown

        self.orig_hero_row = hero_row
        self.orig_hero_col = hero_col
        self.orig_hero_dir = hero_dir

        if decision_maker is None:
            self.decision_maker = RandomDecisionMaker.auto_init()
        else:
            self.decision_maker = decision_maker

        self.original_markers = original_markers
        # original markers show the places where it's necessary that there were or
        # were no markers present: 0 shows it does not matter, 1 shows the number of
        # present markers, -1 shows that we are sure we don't have any markers there

    @classmethod
    def empty_init(cls, rows: int, cols: int, decision_maker: DecisionMaker = None):
        return cls(rows, cols, None, None, None, np.zeros((rows, cols), dtype=int),
                   np.zeros((rows, cols), dtype=int),
                   np.ones((rows, cols), dtype=int),
                   decision_maker,
                   np.zeros((rows, cols), dtype=float))

    @classmethod
    def hero_position_only_init(cls, rows: int, cols: int,
                                hero_row: Optional[int], hero_col: Optional[int],
                                hero_dir: Optional[Direction],
                                decision_maker: DecisionMaker = None):
        return cls(rows, cols, hero_row, hero_col, hero_dir,
                   np.zeros((rows, cols), dtype=int),
                   np.zeros((rows, cols), dtype=int),
                   np.ones((rows, cols), dtype=int),
                   decision_maker,
                   np.zeros((rows, cols), dtype=float))

    @classmethod
    def parse_json(cls, json_dict, decision_maker: DecisionMaker = None):
        rows = json_dict['rows']
        cols = json_dict['cols']

        hero_row = None
        hero_col = None
        hero_dir = None

        if json_dict['hero'] != "":
            hero_pos = json_dict['hero'].split(':')

            hero_row = int(hero_pos[0])
            hero_col = int(hero_pos[1])
            hero_dir = str(hero_pos[2])

        original_markers = np.zeros((rows, cols), dtype=float)

        blocked = np.zeros((rows, cols), dtype=int)
        if json_dict['blocked'] != "":
            for x in json_dict['blocked'].split(' '):
                pos = x.split(':')
                blocked[int(pos[0])][int(pos[1])] = 1

        markers = np.zeros((rows, cols), dtype=int)
        if json_dict['markers'] != "":
            for x in json_dict['markers'].split(' '):
                pos = x.split(':')
                markers[int(pos[0])][int(pos[1])] = int(pos[2])

        unknown = np.zeros((rows, cols), dtype=int)
        if json_dict['unknown'] != "":
            for x in json_dict['unknown'].split(' '):
                pos = x.split(':')
                assert blocked[int(pos[0])][int(pos[1])] == 0
                assert markers[int(pos[0])][int(pos[1])] == 0
                assert not (hero_row == int(pos[0]) and hero_col == int(pos[1]))
                unknown[int(pos[0])][int(pos[1])] = 1

        for row in range(rows):
            for col in range(cols):
                if unknown[row][col] == 0:
                    if blocked[row][col] == 1:
                        pass
                    elif markers[row][col] > 0:
                        original_markers[row][col] = markers[row][col]
                    elif not (hero_row == row and hero_col == col):
                        original_markers[row][col] = -0.25

        return cls(rows, cols, hero_row, hero_col, hero_dir, blocked, markers,
                   unknown, decision_maker, original_markers)

    def set_decision_maker(self, decision_maker: DecisionMaker):
        self.decision_maker = decision_maker

    def get_hero_pos(self):
        # if self.hero_row is None:
        #     self.hero_row = self.decision_maker.pick_int(0, self.rows)
        #     self.orig_hero_row = self.hero_row
        # if self.hero_col is None:
        #     self.hero_col = self.decision_maker.pick_int(0, self.cols)
        #     self.orig_hero_col = self.hero_col
        if self.heroRow is None or self.heroCol is None:
            possible_positions = [get_position_from_quadrant(Quadrant.center,
                                                             self.rows, self.cols),
                                  get_position_from_quadrant(Quadrant.bottom_left,
                                                             self.rows, self.cols),
                                  get_position_from_quadrant(Quadrant.top_left,
                                                             self.rows, self.cols),
                                  get_position_from_quadrant(Quadrant.bottom_right,
                                                             self.rows, self.cols),
                                  get_position_from_quadrant(Quadrant.top_right,
                                                             self.rows, self.cols)]
            self.heroRow, self.heroCol = possible_positions[
                self.decision_maker.pick_int(0, len(possible_positions), 'pos')]
            self.orig_hero_row = self.heroRow
            self.orig_hero_col = self.heroCol

        if self.blocked[self.heroRow][self.heroCol]:
            self.heroRow = None
            self.heroCol = None
            return self.get_hero_pos()
        self.unknown[self.heroRow][self.heroCol] = 0
        return self.heroRow, self.heroCol

    def get_hero_dir(self):
        if self.heroDir is None:
            int_dir = self.decision_maker.pick_int(1, 5, 'dir')
            self.heroDir = self.int_to_hero_dir(int_dir)
            self.orig_hero_dir = self.heroDir
        return self.heroDir

    def hero_dir_to_int(self, hero_dir=None):
        if hero_dir is None:
            hero_dir = self.get_hero_dir()
        if hero_dir == Direction.north: return 1
        if hero_dir == Direction.east: return 2
        if hero_dir == Direction.south: return 3
        if hero_dir == Direction.west: return 4
        raise Exception('invalid dir')

    @staticmethod
    def int_to_hero_dir(int_dir):
        if int_dir == 1: return Direction.north
        if int_dir == 2: return Direction.east
        if int_dir == 3: return Direction.south
        if int_dir == 4: return Direction.west
        raise Exception('invalid dir')

    def hero_at_pos(self, r, c):
        hero_row, hero_col = self.get_hero_pos()
        if hero_row != r: return False
        if hero_col != c: return False
        return True

    def orig_hero_at_pos(self, r, c):
        if self.orig_hero_row != r: return False
        if self.orig_hero_col != c: return False
        return True

    def isCrashed(self):
        return self.crashed

    def decide_unknown(self, r, c):
        if self.crashed: return
        if r < 0 or c < 0:
            return
        if r >= self.rows or c >= self.cols:
            return
        if self.unknown[r][c] != 1:
            return
        self.unknown[r][c] = 0
        return self.decision_maker.binary_decision()

    def is_clear_stochastic(self, r, c):
        if self.crashed: return
        if r < 0 or c < 0:
            return False
        if r >= self.rows or c >= self.cols:
            return False
        if self.unknown[r][c]:
            decision = self.decide_unknown(r, c)
            if not decision:
                self.blocked[r][c] = 1
                return False
            else:
                self.blocked[r][c] = 0
                return True
        else:
            return self.blocked[r][c] == 0

    def is_clear(self, r, c):
        if self.crashed: return
        if r < 0 or c < 0:
            return False
        if r >= self.rows or c >= self.cols:
            return False
        if self.blocked[r][c]:
            return False
        self.unknown[r][c] = 0
        return True

    def frontIsClear(self):
        if self.crashed: return
        hero_row, hero_col = self.get_hero_pos()
        hero_dir = self.get_hero_dir()
        if hero_dir == Direction.north:
            return self.is_clear_stochastic(hero_row + 1, hero_col)
        if hero_dir == Direction.east:
            return self.is_clear_stochastic(hero_row, hero_col + 1)
        if hero_dir == Direction.south:
            return self.is_clear_stochastic(hero_row - 1, hero_col)
        if hero_dir == Direction.west:
            return self.is_clear_stochastic(hero_row, hero_col - 1)
        raise Exception('invalid dir')

    def leftIsClear(self):
        if self.crashed: return
        hero_row, hero_col = self.get_hero_pos()
        hero_dir = self.get_hero_dir()
        if hero_dir == Direction.north:
            return self.is_clear_stochastic(hero_row, hero_col - 1)
        if hero_dir == Direction.east:
            return self.is_clear_stochastic(hero_row + 1, hero_col)
        if hero_dir == Direction.south:
            return self.is_clear_stochastic(hero_row, hero_col + 1)
        if hero_dir == Direction.west:
            return self.is_clear_stochastic(hero_row - 1, hero_col)
        raise Exception('invalid dir')

    def rightIsClear(self):
        if self.crashed: return
        hero_row, hero_col = self.get_hero_pos()
        hero_dir = self.get_hero_dir()
        if hero_dir == Direction.north:
            return self.is_clear_stochastic(hero_row, hero_col + 1)
        if hero_dir == Direction.east:
            return self.is_clear_stochastic(hero_row - 1, hero_col)
        if hero_dir == Direction.south:
            return self.is_clear_stochastic(hero_row, hero_col - 1)
        if hero_dir == Direction.west:
            return self.is_clear_stochastic(hero_row + 1, hero_col)
        raise Exception('invalid dir')

    def markersPresent(self):
        if self.crashed: return

        hero_row, hero_col = self.get_hero_pos()

        if self.markers[hero_row][hero_col] > 0:
            return True
        elif self.original_markers[hero_row][hero_col] < 0:
            return False

        if self.decision_maker.binary_decision():
            self.markers[hero_row][hero_col] = 1
            if self.original_markers[hero_row][hero_col] == 0:
                self.original_markers[hero_row][hero_col] = 0.5
            return True
        else:
            self.markers[hero_row][hero_col] = 0
            if self.original_markers[hero_row][hero_col] == 0:
                self.original_markers[hero_row][hero_col] = -0.25
            else:
                self.original_markers[hero_row][hero_col] = -self.original_markers[
                    hero_row][hero_col]
            return False

    def pickMarker(self):
        if self.crashed: return
        hero_row, hero_col = self.get_hero_pos()
        if self.original_markers[hero_row][hero_col] < 0:
            if self.markers[hero_row][hero_col] == 0:
                self.crashed = True
                return
            self.markers[hero_row][hero_col] -= 1
        else:
            self.original_markers[hero_row][hero_col] += 1
            self.markers[hero_row][hero_col] = max(0,
                                                   self.markers[hero_row][hero_col] - 1)

    def putMarker(self):
        if self.crashed: return
        hero_row, hero_col = self.get_hero_pos()
        if self.original_markers[hero_row][hero_col] == 0:
            self.original_markers[hero_row][hero_col] = -0.25
        self.markers[hero_row][hero_col] += 1

    def run(self):
        if self.heroRow is None or self.heroCol is None or self.heroDir is None:
            tuple_idx = self.decision_maker.pick_int(0, 19, 'loc')
            quad, dir = idx2quad_dir[tuple_idx]
            pos = get_position_from_quadrant(quad, self.rows, self.cols)
            self.heroDir = dir
            self.heroRow = pos[0]
            self.heroCol = pos[1]

            self.orig_hero_row = self.heroRow
            self.orig_hero_col = self.heroCol
            self.orig_hero_dir = self.heroDir

            if self.blocked[self.heroRow][self.heroCol]:
                self.heroRow = None
                self.heroCol = None
                self.heroDir = None
                self.run()

            self.unknown[self.heroRow][self.heroCol] = 0

        # self.get_hero_pos()
        # self.get_hero_dir()

    def move(self):
        if self.crashed: return
        new_row, new_col = self.get_hero_pos()
        # if self.original_markers[new_row][new_col] == 0 and self.markers[new_row][
        #     new_col] == 1:
        #     self.original_markers[new_row][new_col] = 1
        hero_dir = self.get_hero_dir()
        if hero_dir == Direction.north:
            new_row += 1
        if hero_dir == Direction.east:
            new_col += 1
        if hero_dir == Direction.south:
            new_row -= 1
        if hero_dir == Direction.west:
            new_col -= 1
        if not self.is_clear(new_row, new_col):
            self.crashed = True
        if not self.crashed:
            self.heroRow = new_row
            self.heroCol = new_col

    def turnLeft(self):
        if self.crashed: return
        hero_dir = self.get_hero_dir()
        if hero_dir == Direction.north:
            self.heroDir = Direction.west
        if hero_dir == Direction.east:
            self.heroDir = Direction.north
        if hero_dir == Direction.south:
            self.heroDir = Direction.east
        if hero_dir == Direction.west:
            self.heroDir = Direction.south

    def turnRight(self):
        if self.crashed: return
        hero_dir = self.get_hero_dir()
        if hero_dir == Direction.north:
            self.heroDir = Direction.east
        if hero_dir == Direction.east:
            self.heroDir = Direction.south
        if hero_dir == Direction.south:
            self.heroDir = Direction.west
        if hero_dir == Direction.west:
            self.heroDir = Direction.north

    def execute_action(self, action_string):
        action_func = getattr(self, action_string)
        action_func()

    def get_hero_char(self):
        if self.heroDir == Direction.north:
            return '^'
        if self.heroDir == Direction.east:
            return '>'
        if self.heroDir == Direction.south:
            return 'v'
        if self.heroDir == Direction.west:
            return '<'
        if self.heroDir == Direction.any:
            return '+'
        raise Exception('invalid dir')

    def __str__(self):
        world_str = '-' * self.cols + '\n'
        # world_str += str(self.heroRow) + ', ' + str(self.heroCol) + '\n'
        if self.crashed: world_str += 'CRASHED\n'
        if self.heroRow is None or self.heroCol is None:
            return 'HERO NOT PLACED YET'
        for r in range(self.rows - 1, -1, -1):
            row_str = '|'
            for c in range(0, self.cols):
                if self.blocked[r][c] == 1:
                    row_str += '*'
                elif self.hero_at_pos(r, c):
                    row_str += self.get_hero_char()
                elif self.markers[r][c] > 0:
                    numMarkers = int(self.markers[r][c])
                    if numMarkers > 9:
                        row_str += 'M'
                    else:
                        row_str += str(numMarkers)
                elif self.unknown[r][c]:
                    row_str += '?'
                else:
                    row_str += ' '
            world_str += row_str + '|'
            if r != 0: world_str += '\n'
        world_str += '-' * self.cols
        return world_str

    def draw(self):
        worldStr = ''.join(f'{Tcolors.GRAY}▃{Tcolors.ENDC}'
                           for _ in range(self.cols + 2)) + "\n"
        for r in range(self.rows - 1, -1, -1):
            rowStr = f'{Tcolors.GRAY}▌{Tcolors.ENDC}'
            for c in range(0, self.cols):
                if self.blocked[r][c] == 1:
                    rowStr += f'{Tcolors.GRAY}#{Tcolors.ENDC}'
                elif self.get_hero_pos() == (r, c):
                    rowStr += f"{Tcolors.OKGREEN}{self.get_hero_char()}{Tcolors.ENDC}"
                elif self.markers[r][c] > 0:
                    numMarkers = int(self.markers[r][c])
                    if numMarkers > 9:
                        rowStr += f"{Tcolors.WARNING}M{Tcolors.ENDC}"
                    else:
                        rowStr += f"{Tcolors.WARNING}{numMarkers}{Tcolors.ENDC}"
                elif self.unknown[r][c]:
                    rowStr += f"{Tcolors.FAIL}?{Tcolors.ENDC}"
                else:
                    rowStr += f"."
            worldStr += rowStr + f'{Tcolors.GRAY}▐{Tcolors.ENDC}'
            worldStr += '\n'
        worldStr += ''.join(f'{Tcolors.GRAY}▀{Tcolors.ENDC}' for _ in range(
            self.cols + 2)) + "\n"
        return worldStr

    def to_json(self):
        obj = {'rows': self.rows, 'cols': self.cols}

        if self.crashed:
            obj['crashed'] = True
            return obj

        obj['crashed'] = False

        markers = []
        blocked = []
        hero = []
        original_markers = []
        unknown = []
        orig_hero = []
        for r in range(self.rows - 1, -1, -1):
            for c in range(0, self.cols):
                if self.blocked[r][c] == 1:
                    blocked.append("{0}:{1}".format(r, c))
                if self.hero_at_pos(r, c):
                    hero.append("{0}:{1}:{2}".format(r, c, self.heroDir))
                if self.orig_hero_at_pos(r, c):
                    orig_hero.append("{0}:{1}:{2}".format(r, c, self.orig_hero_dir))
                if self.markers[r][c] > 0:
                    markers.append(
                        "{0}:{1}:{2}".format(r, c, int(self.markers[r][c])))
                if self.original_markers[r][c] > 0:
                    original_markers.append("{0}:{1}:{2}".
                                            format(r, c,
                                                   float(self.original_markers[r][c])))
                if self.unknown[r][c] == 1:
                    unknown.append("{0}:{1}".format(r, c))

        obj['markers'] = " ".join(markers)
        obj['blocked'] = " ".join(blocked)
        obj['hero'] = " ".join(hero)
        obj['original_markers'] = " ".join(original_markers)
        obj['unknown'] = " ".join(unknown)
        obj['orig_hero'] = " ".join(orig_hero)

        return obj

    def to_tensor(self, padding):
        # Dimensions:
        # pos + dir --> 4
        # orig pos + dir --> 4
        # blocked --> 1
        # unknown --> 1
        # markers --> 10
        # orig markers --> 10 + 1 + 1

        depth = 4 + 4 + 1 + 1 + 10 + 10 + 1 + 1
        tensor = torch.FloatTensor(depth, padding, padding).zero_()

        # IMPORTANT: this is different from the tensor representation of the World,
        # as an 'full-one' indicates a HERO DIRECTION NOT SET YET
        # while it represents and 'any' in the World
        if self.heroDir:
            tensor[self.hero_dir_to_int() - 1][self.heroRow][
                self.heroCol] = 1
        else:
            tensor[0:4][:, self.heroRow, self.heroCol] = 1

        if self.orig_hero_dir:
            tensor[self.hero_dir_to_int() + 4 - 1, self.orig_hero_row,
                   self.orig_hero_col] = 1

        for r in range(self.rows):
            for c in range(self.cols):
                tensor[4 + 4][r][c] = self.blocked[r][c]
                nb_markers = int(self.markers[r][c])
                if nb_markers > 0:
                    tensor[4 + 4 + 1 + nb_markers][r][c] = 1
                if self.unknown[r][c]:
                    tensor[4 + 4 + 1 + 10][r][c] = 1
                orig_nb_markers = float(self.original_markers[r][c])
                if orig_nb_markers != 0:
                    if orig_nb_markers != -0.25:
                        tensor[4 + 4 + 1 + 10 + int(orig_nb_markers)][r][c] = 1
                if orig_nb_markers < 0:
                    tensor[4 + 4 + 1 + 10 + 10 + 1][r][c] = 1
                if orig_nb_markers - int(orig_nb_markers) == 0.5:
                    tensor[4 + 4 + 1 + 10 + 10 + 1 + 1][r][c] = 1

        return tensor
