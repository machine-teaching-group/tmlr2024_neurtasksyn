import copy
from unittest.mock import MagicMock

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation

from src.emulator.code import Code
from src.emulator.fast_emulator import FastEmulator
from src.emulator.world import World
from src.symexecution.decision_makers import RandomDecisionMaker
from src.symexecution.symworld import SymWorld
from src.symexecution.utils.enums import Direction


def unknown_only_closing(blocked: np.ndarray, unknowns: np.ndarray) -> np.ndarray:
    shape = np.ones((3, 3), dtype=int)
    result = binary_dilation(blocked, iterations=1,
                             border_value=0)
    result = binary_erosion(result, shape, iterations=1,
                            border_value=1)
    result = unknowns & result
    return result


class PostProcessor:

    @staticmethod
    def symworld_to_world(symworl):
        pass

    @staticmethod
    def handle_initial_markers(symworld):
        initial_markers = symworld.original_markers
        initial_markers = np.abs(initial_markers)
        initial_markers = np.where(initial_markers == 0.5, 1, initial_markers)
        initial_markers = initial_markers.astype(int)
        symworld.original_markers = initial_markers


class EmptySpacePostProcessor(PostProcessor):
    @staticmethod
    def symworld_to_world(symworld: SymWorld) -> (World, World):
        super(EmptySpacePostProcessor, EmptySpacePostProcessor). \
            handle_initial_markers(symworld)
        inp_world = World(symworld.rows, symworld.cols,
                          symworld.orig_hero_row, symworld.orig_hero_col,
                          symworld.orig_hero_dir,
                          symworld.blocked,
                          symworld.original_markers)
        out_world = World(symworld.rows, symworld.cols,
                          symworld.heroRow, symworld.heroCol,
                          symworld.heroDir,
                          symworld.blocked,
                          symworld.markers)
        return inp_world, out_world


class BlockedPostProcessor(PostProcessor):
    @staticmethod
    def symworld_to_world(symworld: SymWorld) -> (World, World):
        super(BlockedPostProcessor, BlockedPostProcessor). \
            handle_initial_markers(symworld)
        inp_world = World(symworld.rows, symworld.cols,
                          symworld.orig_hero_row, symworld.orig_hero_col,
                          symworld.orig_hero_dir,
                          symworld.blocked + symworld.unknown,
                          symworld.original_markers)
        out_world = World(symworld.rows, symworld.cols,
                          symworld.heroRow, symworld.heroCol,
                          symworld.heroDir,
                          symworld.blocked + symworld.unknown,
                          symworld.markers)
        return inp_world, out_world


class MorphologicalPostProcessor(PostProcessor):
    @staticmethod
    def symworld_to_world(symworld: SymWorld) -> (World, World):
        super(MorphologicalPostProcessor, MorphologicalPostProcessor). \
            handle_initial_markers(symworld)
        aux = copy.deepcopy(symworld.unknown)
        aux = np.where(aux == 1, np.random.randint(0, 2, (symworld.rows,
                                                          symworld.cols)), 0)
        aux = unknown_only_closing(aux + symworld.blocked, symworld.unknown)
        inp_world = World(symworld.rows, symworld.cols,
                          symworld.orig_hero_row, symworld.orig_hero_col,
                          symworld.orig_hero_dir,
                          aux + symworld.blocked,
                          symworld.original_markers)
        out_world = World(symworld.rows, symworld.cols,
                          symworld.heroRow, symworld.heroCol,
                          symworld.heroDir,
                          aux + symworld.blocked,
                          symworld.markers)
        return inp_world, out_world


class CopyPostProcessor(PostProcessor):
    @staticmethod
    def symworld_to_world(symworld: SymWorld,
                          pregrid_to_copy: World,
                          postgrid_to_copy: World) -> (World, World):
        super(CopyPostProcessor, CopyPostProcessor). \
            handle_initial_markers(symworld)
        aux_blocked = copy.deepcopy(pregrid_to_copy.blocked)
        mask = symworld.unknown == 0
        aux_blocked[mask] = 0

        aux_post_markers = copy.deepcopy(postgrid_to_copy.markers)
        mask = symworld.unknown == 0
        aux_post_markers[mask] = 0

        aux_pre_markers = copy.deepcopy(pregrid_to_copy.markers)
        mask = symworld.unknown == 0
        aux_pre_markers[mask] = 0

        inp_world = World(symworld.rows, symworld.cols,
                          symworld.orig_hero_row, symworld.orig_hero_col,
                          symworld.orig_hero_dir,
                          aux_blocked + symworld.blocked,
                          aux_pre_markers + symworld.original_markers)
        out_world = World(symworld.rows, symworld.cols,
                          symworld.heroRow, symworld.heroCol,
                          symworld.heroDir,
                          aux_blocked + symworld.blocked,
                          aux_post_markers + symworld.markers)

        if inp_world.heroRow is None:
            inp_world.heroRow = pregrid_to_copy.heroRow
        if inp_world.heroCol is None:
            inp_world.heroCol = pregrid_to_copy.heroCol

        if out_world.heroRow is None:
            out_world.heroRow = postgrid_to_copy.heroRow
        if out_world.heroCol is None:
            out_world.heroCol = postgrid_to_copy.heroCol

        if inp_world.heroDir is None:
            inp_world.heroDir = pregrid_to_copy.heroDir
        if out_world.heroDir is None:
            out_world.heroDir = postgrid_to_copy.heroDir

        return inp_world, out_world
