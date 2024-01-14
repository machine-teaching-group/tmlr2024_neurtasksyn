from typing import Tuple, List

import numpy as np

from src.emulator.code import Code
from src.emulator.fast_emulator import FastEmulator, EmuResult
from src.emulator.task import Task
from src.emulator.world import World


# def count_moves(body: list) -> int:
#     """Count the number of moves in a code snippet."""
#     num_moves = 0
#     for action in body:
#         if action in BASIC_ACTIONS:
#             if action['type'] == 'move':
#                 num_moves += 1
#         elif action['type'] in ['if', 'while', 'repeat']:
#             num_moves += count_moves(action['body'])
#         elif action['type'] == 'ifElse':
#             num_moves += count_moves(action['ifBody'])
#             num_moves += count_moves(action['elseBody'])
#
#     return num_moves
#
#
# def get_number_of_moves(code_json: dict) -> int:
#     """Get the number of moves in a code snippet."""
#     return count_moves(code_json['run'])


def count_moves(code: Code, world: World) -> int:
    emulator = FastEmulator()
    result = emulator.emulate(code, world)
    return len(list(filter(lambda x: x == 'move', result.actions)))


def count_turns(code: Code, world: World) -> int:
    emulator = FastEmulator()
    result = emulator.emulate(code, world)
    return len(list(filter(lambda x: x in ['turnLeft', 'turnRight'], result.actions)))


def count_segments_from_visited(visited: np.ndarray) -> Tuple[int, int]:
    saturated = np.zeros((visited.shape[0] + 2, visited.shape[1] + 2))
    saturated[1:-1, 1:-1] = visited.copy()
    saturated[saturated > 1] = 1

    # verticals
    m = np.diff(saturated, 1, 0)  # get discrete difference along the 0-axis
    m = np.argwhere(m != 0)  # get indices where value is not zero
    m = m[np.lexsort(m.T)]  # sort indices first 1-column then 0-column
    m[::2, 0] += 1  #

    if m.shape == (7, 2):
        print(m)

    distance = m[1::2, :] - m[::2]
    segments = np.argwhere(distance[:, 0] >= 2).shape[0]
    long_segments = np.argwhere(distance[:, 0] >= 4).shape[0]

    # horizontals
    m = np.diff(saturated, 1, 1, append=np.zeros((saturated.shape[0], 1)))
    m = np.argwhere(m != 0)
    m[::2, 1] += 1

    distance = m[1::2, :] - m[::2]
    segments += np.argwhere(distance[:, 1] >= 2).shape[0]
    long_segments += np.argwhere(distance[:, 1] >= 4).shape[0]

    return segments, long_segments


def count_segments(code: Code, world: World) -> Tuple[int, int]:
    emulator = FastEmulator()
    result = emulator.emulate(code, world)

    return count_segments_from_visited(result.visited)


def count_pick_markers(code: Code, world: World) -> int:
    emulator = FastEmulator()
    result = emulator.emulate(code, world)
    return len(list(filter(lambda x: x == 'pickMarker', result.actions)))


def count_put_markers(code: Code, world: World) -> int:
    emulator = FastEmulator()
    result = emulator.emulate(code, world)
    return len(list(filter(lambda x: x == 'putMarker', result.actions)))


def count_all(code: Code, world: World) -> Tuple[int, int, int, int, int, int]:
    emulator = FastEmulator()
    result = emulator.emulate(code, world)
    moves, turns, pick_markers, put_markers = 0, 0, 0, 0

    for i in range(0, len(result.actions)):
        if result.actions[i] == 'move':
            moves += 1
        elif result.actions[i] in ['turnLeft', 'turnRight']:
            turns += 1
        elif result.actions[i] == 'pickMarker':
            pick_markers += 1
        elif result.actions[i] == 'putMarker':
            put_markers += 1

    segments, long_segments = count_segments_from_visited(result.visited)

    return moves, turns, segments, long_segments, pick_markers, put_markers


def compute_codeworld_quality(code: Code, world: World):
    """Compute the quality of a code snippet for one world."""
    n = max(world.rows, world.cols)
    no_moves, no_turns, no_segments, \
        no_long_segments, no_pick_markers, no_put_markers = \
        count_all(code, world)

    no_moves /= (2 * n)
    no_turns /= n
    no_segments /= (n / 2)
    no_long_segments /= (n / 3)
    no_pick_markers /= n
    no_put_markers /= n

    quality = 0.75 * 0.25 * (no_moves +
                             no_turns +
                             no_segments +
                             no_long_segments) + \
              0.25 * 0.5 * (no_pick_markers +
                            no_put_markers)

    info = {"moves": no_moves,
            "turns": no_turns,
            "segments": no_segments,
            "long_segments": no_long_segments,
            "pick_markers": no_pick_markers,
            "put_markers": no_put_markers}

    return quality, info


def compute_codetask_quality(code: Code, task: Task):
    """Compute the quality of a code snippet for a task."""
    pregrid_quality = 0
    pregrid_info = []
    for world in task.pregrids:
        quality, info = compute_codeworld_quality(code, world)
        pregrid_quality += quality
        pregrid_info.append(info)
    pregrid_quality /= task.num_examples
    return pregrid_quality, pregrid_info


def count_codeworld_quality_from_emulator_actions_and_visitation(actions: list,
                                                                 visited: np.ndarray):
    count_turns, temp2 = 0, 1
    moves, turns, pick_markers, put_markers = 0, 0, 0, 0

    if len(actions) == 0:
        return moves, turns, 0, 0, pick_markers, put_markers, count_turns

    for act in actions:
        if act == 'move':
            moves += 1
        elif act == 'turnLeft' or act == 'turnRight':
            turns += 1
        elif act == 'pickMarker':
            pick_markers += 1
        elif act == 'putMarker':
            put_markers += 1

    aux_actions = list(filter(lambda x: x not in ['pickMarker', 'putMarker'], actions))
    for i in range(1, len(aux_actions)):

        if aux_actions[i] == 'turnLeft' or aux_actions[i] == 'turnRight':
            if aux_actions[i] == aux_actions[i - 1]:
                temp2 += 1
            else:
                if temp2 >= 3:
                    count_turns += 1

                temp2 = 1

    if temp2 >= 3:
        count_turns += 1

    segments, long_segments = count_segments_from_visited(visited)

    return moves, turns, segments, long_segments, pick_markers, put_markers, count_turns


def count_codeworld_quality_from_emulator_actions(actions: list):
    count_turns, temp2 = 0, 1
    count_simple, count_long, temp = 0, 0, 1
    moves, turns, pick_markers, put_markers = 0, 0, 0, 0

    if len(actions) == 0:
        return moves, turns, count_simple, count_long, pick_markers, put_markers, count_turns

    for act in actions:
        if act == 'move':
            moves += 1
        elif act == 'turnLeft' or act == 'turnRight':
            turns += 1
        elif act == 'pickMarker':
            pick_markers += 1
        elif act == 'putMarker':
            put_markers += 1

    aux_actions = list(filter(lambda x: x not in ['pickMarker', 'putMarker'], actions))
    for i in range(1, len(aux_actions)):
        if aux_actions[i] == 'move':
            if aux_actions[i] == aux_actions[i - 1]:
                temp += 1
            else:
                if temp >= 3:
                    count_simple += 1
                    if temp >= 5:
                        count_long += 1

                temp = 1

        elif aux_actions[i] == 'turnLeft' or aux_actions[i] == 'turnRight':
            if aux_actions[i] == aux_actions[i - 1]:
                temp2 += 1
            else:
                if temp2 >= 3:
                    count_turns += 1

                temp2 = 1

    if temp2 >= 3:
        count_turns += 1

    if temp >= 3:
        count_simple += 1
        if temp >= 5:
            count_long += 1

    return moves, turns, count_simple, count_long, pick_markers, put_markers, count_turns


def compute_karel_codeworld_quality_from_emulator_actions_and_visitation(actions: list,
                                                                         visited: np.ndarray,
                                                                         n: int):
    """Compute the quality of a code snippet for one world."""
    no_moves, no_turns, no_segments, \
        no_long_segments, no_pick_markers, no_put_markers, no_turn_segments = \
        count_codeworld_quality_from_emulator_actions_and_visitation(actions, visited)

    no_moves /= (2 * n)
    no_turns /= n
    no_segments /= (n / 2)
    no_long_segments /= (n / 3)
    no_pick_markers /= n
    no_put_markers /= n

    no_turn_segments /= (n / 2)

    no_moves = min(no_moves, 1)
    no_turns = min(no_turns, 1)
    no_segments = min(no_segments, 1)
    no_long_segments = min(no_long_segments, 1)
    no_pick_markers = min(no_pick_markers, 1)
    no_put_markers = min(no_put_markers, 1)

    score_turn_segments = 1 - min(no_turn_segments, 1)

    quality = 0.5 * 0.25 * (no_moves +
                            no_turns +
                            no_segments +
                            no_long_segments) + \
              0.25 * 0.5 * (no_pick_markers +
                            no_put_markers) + \
              0.25 * score_turn_segments

    info = {"moves": no_moves,
            "turns": no_turns,
            "segments": no_segments,
            "long_segments": no_long_segments,
            "pick_markers": no_pick_markers,
            "put_markers": no_put_markers,
            "turn_segments": no_turn_segments,
            "quality": quality}

    return quality, info


def compute_karel_codeworld_quality_from_emulator_actions(actions: list,
                                                          n: int):
    """Compute the quality of a code snippet for one world."""
    no_moves, no_turns, no_segments, \
        no_long_segments, no_pick_markers, no_put_markers, no_turn_segments = \
        count_codeworld_quality_from_emulator_actions(actions)

    no_moves /= (2 * n)
    no_turns /= n
    no_segments /= (n / 2)
    no_long_segments /= (n / 3)
    no_pick_markers /= n
    no_put_markers /= n

    no_turn_segments /= (n / 2)

    no_moves = min(no_moves, 1)
    no_turns = min(no_turns, 1)
    no_segments = min(no_segments, 1)
    no_long_segments = min(no_long_segments, 1)
    no_pick_markers = min(no_pick_markers, 1)
    no_put_markers = min(no_put_markers, 1)

    score_turn_segments = 1 - min(no_turn_segments, 1)

    quality = 0.5 * 0.25 * (no_moves +
                            no_turns +
                            no_segments +
                            no_long_segments) + \
              0.25 * 0.5 * (no_pick_markers +
                            no_put_markers) + \
              0.25 * score_turn_segments

    info = {"moves": no_moves,
            "turns": no_turns,
            "segments": no_segments,
            "long_segments": no_long_segments,
            "pick_markers": no_pick_markers,
            "put_markers": no_put_markers,
            "turn_segments": no_turn_segments,
            "quality": quality}

    return quality, info


def compute_hoc_codeworld_quality_from_emulator_actions(actions: list,
                                                        n: int):
    """Compute the quality of a code snippet for one world."""
    no_moves, no_turns, no_segments, \
        no_long_segments, no_pick_markers, no_put_markers, no_turn_segments = \
        count_codeworld_quality_from_emulator_actions(actions)

    no_moves /= (2 * n)
    no_turns /= n
    no_segments /= (n / 2)
    no_long_segments /= (n / 3)

    no_turn_segments /= (n / 2)

    no_moves = min(no_moves, 1)
    no_turns = min(no_turns, 1)
    no_segments = min(no_segments, 1)
    no_long_segments = min(no_long_segments, 1)

    score_turn_segments = 1 - min(no_turn_segments, 1)

    quality = 0.75 * 0.25 * (no_moves +
                             no_turns +
                             no_segments +
                             no_long_segments) + \
              0.25 * score_turn_segments

    info = {"moves": no_moves,
            "turns": no_turns,
            "segments": no_segments,
            "long_segments": no_long_segments,
            "pick_markers": 0,
            "put_markers": 0,
            "turn_segments": no_turn_segments,
            "quality": quality}

    return quality, info


def compute_hoc_codeworld_quality_from_emulator_actions_and_visitation(actions: list,
                                                                       visited: np.ndarray,
                                                                       n: int,
                                                                       segments_weight=None):
    """Compute the quality of a code snippet for one world."""
    no_moves, no_turns, no_segments, \
        no_long_segments, no_pick_markers, no_put_markers, no_turn_segments = \
        count_codeworld_quality_from_emulator_actions_and_visitation(actions, visited)

    no_moves /= (2 * n)
    no_turns /= n

    if segments_weight is None:
        no_segments /= (n / 2)
        no_long_segments /= (n / 3)

    no_turn_segments /= (n / 2)

    no_moves = min(no_moves, 1)
    no_turns = min(no_turns, 1)
    no_segments = min(no_segments, 1)
    no_long_segments = min(no_long_segments, 1)

    score_turn_segments = 1 - min(no_turn_segments, 1)

    if segments_weight is None:
        quality = 0.75 * 0.25 * (no_moves +
                                 no_turns +
                                 no_segments +
                                 no_long_segments) + \
                  0.25 * score_turn_segments
    else:
        quality = 0.75 * 0.25 * (no_moves +
                                 no_turns +
                                 no_segments * segments_weight +
                                 no_long_segments * segments_weight) + \
                  0.25 * score_turn_segments

    info = {"moves": no_moves,
            "turns": no_turns,
            "segments": no_segments,
            "long_segments": no_long_segments,
            "pick_markers": 0,
            "put_markers": 0,
            "turn_segments": no_turn_segments,
            "quality": quality}

    return quality, info


def compute_codetask_quality_from_executor_result(result: dict, n: List[int],
                                                  type: str,
                                                  segments_weight=None):
    pregrid_quality = 0
    pregrid_info = []
    for n, res in zip(n, result['emulator_result']):
        # if type == 'karel':
        #     quality, info = \
        #         compute_karel_codeworld_quality_from_emulator_actions_and_visitation(
        #             res['execution_info']['actions'], res['execution_info'][
        #                 'visited'], n)
        # elif type == 'hoc':
        quality, info = \
            compute_hoc_codeworld_quality_from_emulator_actions_and_visitation(
                res['execution_info']['actions'], res['execution_info'][
                    'visited'], n, segments_weight=segments_weight)
        # else:
        #     raise ValueError('Invalid type')
        pregrid_quality += quality
        pregrid_info.append(info)
    pregrid_quality /= len(result['emulator_result'])
    return pregrid_quality, pregrid_info


def compute_codetask_quality_from_emulator_result(result: EmuResult, n: int,
                                                  type: str):
    pregrid_info = []
    # if type == 'karel':
    #     quality, info = \
    #         compute_karel_codeworld_quality_from_emulator_actions_and_visitation(
    #             result.actions, result.visited, n)
    # elif type == 'hoc':
    quality, info = \
        compute_hoc_codeworld_quality_from_emulator_actions_and_visitation(
            result.actions, result.visited, n)
    # else:
    #     raise ValueError('Invalid type')
    pregrid_info.append(info)
    return quality, pregrid_info


def compute_codetask_quality_from_actions(actions: list, n: int, type: str):
    pregrid_info = []
    if type == 'karel':
        quality, info = \
            compute_karel_codeworld_quality_from_emulator_actions(
                actions, n)
    elif type == 'hoc':
        quality, info = \
            compute_hoc_codeworld_quality_from_emulator_actions(
                actions, n)
    else:
        raise ValueError('Invalid type')
    pregrid_info.append(info)
    return quality, pregrid_info


def compute_visitation_quality_from_emulator_result(result: EmuResult):
    ones = (result.visited == 1).sum()
    others = (result.visited > 1).sum()

    return ones / (ones + others)


def compute_visitation_quality_from_executor_result(result: dict):
    lst = []
    for res in result['emulator_result']:
        ones = (res['execution_info']['visited'] == 1).sum()
        others = (res['execution_info']['visited'] > 1).sum()
        lst.append(ones / (ones + others))

    return lst


if __name__ == '__main__':
    code_json = {'program_type': 'karel',
                 'program_json': {'type': 'run',
                                  "body":
                                      [{"type": "while",
                                        "condition": "noMarkersPresent",
                                        "body": [{"type": "putMarker"},
                                                 {"type": "putMarker"},
                                                 {"type": "move"},
                                                 {"type": "move"},
                                                 {"type": "move"},
                                                 {"type": "move"},
                                                 {"type": "move"},
                                                 {"type": "turnLeft"}]
                                        },
                                       {"type": "turnLeft"},
                                       {"type": "while", "condition": "markersPresent",
                                        "body": [{"type": "pickMarker"},
                                                 {"type": "pickMarker"},
                                                 {"type": "move"},
                                                 {"type": "move"},
                                                 {"type": "move"},
                                                 {"type": "move"},
                                                 {"type": "move"},
                                                 {"type": "turnRight"}]
                                        },
                                       ]}}

    world_json = {
        'rows': 15,
        'cols': 15,
        'markers': '',
        'blocked': '',
        'hero': '6:6:north',
    }

    world = World.parseJson(world_json)
    code = Code.parse_json(code_json)

    emulator = FastEmulator(1000, 1000)
    emulator_result = emulator.emulate(code, world)

    print(compute_codetask_quality_from_emulator_result(emulator_result, 15, 'karel'))
    print(emulator_result.inpgrid.draw())
    print(emulator_result.outgrid.draw())
