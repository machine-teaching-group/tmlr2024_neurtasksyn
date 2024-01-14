import copy
import itertools
import json

import numpy as np
import torch
from torch import Tensor

from src.codetask_scoring.execquality import \
    count_codeworld_quality_from_emulator_actions_and_visitation
from src.emulator.fast_emulator import FastEmulator
from src.symexecution.utils.enums import Quadrant, idx2quad_dir
from src.symexecution.utils.quadrants import get_position_from_quadrant, \
    get_quadrant_from_position
from src.taskgen.vocabularies import location2idx, decision2idx


def dict_location2short(emulator: FastEmulator):
    dict_loc = json.loads(emulator.current_location)
    type_ = dict_loc['type']
    if type_ in emulator.action_hash:
        return type_
    elif type_ == 'run':
        return 'run'
    elif type_ == 'repeat':
        return f"repeat:{dict_loc['times']}"
    else:
        return f"{type_}:{dict_loc['condition']}"


def get_location_quality_coverage_tensor(emulator: FastEmulator):
    ast = emulator.ast
    world = emulator.state.world
    ticks = emulator.state.ticks

    # The output contains a one-hot encoding of the current location,
    # the values [0,1] for the quality indicators, and the value [0, 1] for coverage
    if ast.type == 'karel':
        tensor = np.zeros(len(location2idx) + 7 + 2)
    elif ast.type == 'hoc':
        tensor = np.zeros(len(location2idx) + 5 + 2)
    else:
        raise ValueError("Unknown ast type")

    loc_shorthand = dict_location2short(emulator=emulator)

    tensor[location2idx[loc_shorthand]] = 1

    no_moves, no_turns, no_segments, \
        no_long_segments, no_pick_markers, no_put_markers, no_turn_segments = \
        count_codeworld_quality_from_emulator_actions_and_visitation(
            emulator.state.actions,
            emulator.visited)

    n = max(world.rows, world.cols)

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

    no_turn_segments = 1 - min(no_turn_segments, 1)

    tensor[len(location2idx)] = no_moves
    tensor[len(location2idx) + 1] = no_turns
    tensor[len(location2idx) + 2] = no_segments
    tensor[len(location2idx) + 3] = no_long_segments
    tensor[len(location2idx) + 4] = no_pick_markers

    if ast.type == 'karel':
        tensor[len(location2idx) + 5] = no_put_markers
        tensor[len(location2idx) + 6] = no_turn_segments

    nb_nodes = ast.total_count
    # Find all unique locations visited from code
    combined_ticks = []
    combined_set = set()
    for tick_task in [ticks]:
        ticks_set = set()
        for tick in tick_task:
            ticks_set.add(str(tick.location))
            combined_set.add(str(tick.location))
        combined_ticks.append(ticks_set)

    coverage = len(combined_set) / nb_nodes

    if ast.type == 'karel':
        tensor[len(location2idx) + 7] = coverage
    elif ast.type == 'hoc':
        tensor[len(location2idx) + 5] = coverage

    ones = (emulator.visited == 1).sum()
    others = (emulator.visited > 1).sum()

    if ones + others == 0:
        vis = 0
    else:
        vis = ones / (ones + others)

    if ast.type == 'karel':
        tensor[len(location2idx) + 8] = vis
    elif ast.type == 'hoc':
        tensor[len(location2idx) + 6] = vis

    return tensor


def features_only_pre_process_input_karel(emulator: FastEmulator):
    tensor = get_location_quality_coverage_tensor(emulator)
    world = emulator.state.world

    initial_markers = world.original_markers
    initial_markers = np.abs(initial_markers)
    initial_markers = np.where(initial_markers == 0.5, 1, initial_markers)
    initial_markers = initial_markers.astype(int)
    initial_markers = np.sum(initial_markers)
    initial_markers = initial_markers / (world.rows * world.cols)
    initial_markers = min(initial_markers, 1)

    tensor = np.append(tensor, initial_markers)

    final_markers = np.sum(world.markers)
    final_markers = final_markers / (world.rows * world.cols)
    final_markers = min(final_markers, 1)

    tensor = np.append(tensor, final_markers)

    walls = np.sum(world.blocked)
    empty = world.cols * world.rows - walls - np.sum(world.unknown)

    walls = walls / (world.rows * world.cols)
    empty = empty / (world.rows * world.cols)

    walls = min(walls, 1)
    empty = min(empty, 1)

    tensor = np.append(tensor, walls)
    tensor = np.append(tensor, empty)

    return {"features": torch.from_numpy(tensor).float()}


def features_only_pre_process_input_hoc(emulator: FastEmulator):
    tensor = get_location_quality_coverage_tensor(emulator)

    world = emulator.state.world

    walls = np.sum(world.blocked)
    empty = world.cols * world.rows - walls - np.sum(world.unknown)

    walls = walls / (world.rows * world.cols)
    empty = empty / (world.rows * world.cols)

    walls = min(walls, 1)
    empty = min(empty, 1)

    tensor = np.append(tensor, walls)
    tensor = np.append(tensor, empty)

    return {"features": torch.from_numpy(tensor).float()}


def features_and_symworld_pre_process_input(emulator: FastEmulator):
    tensor = get_location_quality_coverage_tensor(emulator)

    tensor_dict = {
        "features": torch.from_numpy(tensor).float(),
        "symworld": emulator.state.world.to_tensor(16)
    }

    return tensor_dict


def process_body(action_body, tick_location, action_hash, tick_continue_index=0, type_='body'):
    actions = []
    locations = []
    cont = True
    for idx, action in enumerate(action_body):
        if action['type'] in action_hash:
            if cont:
                actions.append(action['type'])
            locations.append((tick_location + f' {type_}:{tick_continue_index + idx}').strip())
        elif action['type'] == 'repeat':
            repeat_actions = []
            repeat_locations = []
            locations.append((tick_location + f' {type_}:{tick_continue_index + idx}').strip())
            for _ in range(action['times']):
                x, y, cont_rec = process_body(action['body'],
                                              (tick_location + f' {type_}:{tick_continue_index + idx}').strip(),
                                              action_hash)
                if cont:
                    repeat_actions += x
                repeat_locations += y
                if not cont_rec:
                    cont = False
                    break
            actions += repeat_actions
            locations += repeat_locations
        else:
            cont = False
            if action['type'] == 'ifElse':
                locations.append((tick_location + f' {type_}:{tick_continue_index + idx}').strip())
            # break
    return actions, locations, cont


def reach_end(string: str):
    counter = 0
    new_string = ""
    idx = 0
    for c in string:
        if c == '[':
            counter += 1
        elif c == ']':
            counter -= 1
        if counter < 0:
            break
        new_string += c
        idx += 1

    aux = ""
    for c in string[idx:]:
        if c == '}':
            break
        aux += c

    bool_val = True
    if 'times' not in aux:
        bool_val = False

    if new_string.endswith(',{"body":['):
        new_string = new_string[:-9]

    # if 'condition' in new_string:
    #     new_string = new_string.split('condition')[0][:-3]

    return new_string, bool_val


def reach_start(string: str):
    counter = 0
    new_string = ""
    for c in string[::-1]:
        if c == ']':
            counter += 1
        elif c == '[':
            counter -= 1
        if counter < 0:
            break
        new_string += c

    return new_string[::-1]


def process_outer_actions(code_json, current_location, tick_location, action_hash):
    current_location = json.dumps(json.loads(current_location), sort_keys=True)
    upcoming = json.dumps(code_json, sort_keys=True)
    upcoming = upcoming.replace("\n", '')
    upcoming = upcoming.replace("\t", '')
    upcoming = upcoming.replace(" ", '')
    current_location = current_location.replace("\n", '')
    current_location = current_location.replace("\t", '')
    current_location = current_location.replace(" ", '')
    upcoming = upcoming.split(current_location)
    if len(upcoming) > 1:
        before = upcoming[0]
        upcoming = upcoming[1]
    else:
        return []
    before = reach_start(before)
    upcoming, use_before = reach_end(upcoming)
    actions = []
    locations = []
    loc = ' '.join(tick_location.split(' ')[:-1])
    if upcoming != '':
        try:
            if upcoming[-1] == ',':
                upcoming = upcoming[:-1]
            jsn = json.loads('[' + upcoming[1:] + ']')
        except:
            pass
        idx = int(tick_location.split(' ')[-1].split(':')[1])
        x, y, _ = process_body(jsn, loc, action_hash, idx + 1)
        actions += x
        locations += y
    if use_before and before != '':
        x, y, _ = process_body(json.loads('[' + before[:-1] + ']'), loc, action_hash)
        actions += x
        locations += y
    return actions, locations


def find_simple_next_actions(code_json, current_location):
    current_location = json.dumps(json.loads(current_location), sort_keys=True)
    upcoming = json.dumps(code_json, sort_keys=True)
    upcoming = upcoming.replace("\n", '')
    upcoming = upcoming.replace("\t", '')
    upcoming = upcoming.replace(" ", '')
    current_location = current_location.replace("\n", '')
    current_location = current_location.replace("\t", '')
    current_location = current_location.replace(" ", '')
    start_idx = upcoming.find(current_location)
    actions = []
    for idx in range(start_idx, len(upcoming)):
        if upcoming[idx] == ']':
            return json.loads('[' + upcoming[start_idx:idx + 1])
        if upcoming[idx] == 'n':
            if 'condition' in actions:
                return json.loads(
                    '[' + upcoming[start_idx:idx].split('condition')[0] + ']')


def get_next_actions_from_decision_position_code(decision, current_location,
                                                 tick_location,
                                                 code_json,
                                                 action_hash):
    location_dict = json.loads(current_location)
    actions = []
    if tick_location == 'None':
        tick_location = ''
        locations = []
    else:
        locations = [tick_location]
    if location_dict['type'] == 'run':
        x, y, _ = process_body(location_dict['body'], tick_location, action_hash)
        actions += x
        locations += y
    if 'ifElse' in location_dict['type']:
        dec = ('1' in decision and 'no' not in location_dict['condition']) or (
                'no' in location_dict['condition'] and '0' in decision)
        if dec:
            x, y, _ = process_body(location_dict['ifBody'], tick_location, action_hash, type_='ifBody')
            actions += x
            locations += y
        else:
            x, y, _ = process_body(location_dict['elseBody'], tick_location, action_hash, type_='elseBody')
            actions += x
            locations += y
        x, y = process_outer_actions(code_json, current_location, tick_location,
                                     action_hash)
        actions += x
        locations += y
    elif 'if' in location_dict['type'] or 'while' in location_dict[
        'type'] in location_dict['type']:
        dec = ('1' in decision and 'no' not in location_dict['condition']) or (
                'no' in location_dict['condition'] and '0' in decision)
        if dec:
            x, y, _ = process_body(location_dict['body'], tick_location, action_hash)
            actions += x
            locations += y
        if 'if' in location_dict['type'] or (not dec and 'while' in location_dict['type']):
            x, y = process_outer_actions(code_json, current_location, tick_location,
                                         action_hash)
            actions += x
            locations += y
    elif 'repeatUntil' in location_dict['type']:
        if ('0' in decision and 'no' not in location_dict['condition']) or (
                'no' in location_dict['condition'] and '1' in decision):
            x, y, _ = process_body(location_dict['body'], tick_location, action_hash)
            actions += x
            locations += y
        else:
            x, y = process_outer_actions(code_json, current_location,
                                         tick_location,
                                         action_hash)
            actions += x
            locations += y
    elif location_dict['type'] in action_hash:
        print("CE DOBA")
        # aux = json.dumps(code_json, sort_keys=True).replace("\n", '').replace("\t",
        #                                                                       '').replace(
        #     " ", '')
        # curr_loc = json.dumps(location_dict, sort_keys=True).replace("\n", '').replace(
        #     "\t", '').replace(" ", '')
        # idx = aux.find(curr_loc)
        # x, _ = reach_end(aux[idx:])
        # x, y, _ = process_body(json.loads('[' + x + ']'), tick_location, action_hash)
        # actions += x
        # locations += y

    return actions, locations


def recurse_decisions(decisions, current_json, tick_location, action_hash, invert_repeat_until=False):
    actions = []
    locations = []
    if tick_location:
        locations += [tick_location]
    for idx, entry in enumerate(current_json):
        if entry['type'] in action_hash:
            actions.append(entry['type'])
            locations.append(tick_location + f' body:{idx}')
        elif entry['type'] == 'repeat':
            for _ in range(entry['times']):
                x, y, _ = recurse_decisions(decisions, entry['body'], tick_location + f' body:{idx}', action_hash)
                actions += x
                locations += y
                if len(decisions) == 0:
                    return actions
        elif entry['type'] == 'if':
            if len(decisions) > 0:
                dec = decisions.pop(0)
                if dec or ('not' in entry['type'] and not dec):
                    x, y, _ = recurse_decisions(decisions, entry['body'],
                                                tick_location + f' body:{idx}', action_hash)
                    actions += x
                    locations += y
            else:
                return actions
        elif entry['type'] == 'ifElse':
            if len(decisions) > 0:
                dec = decisions.pop(0)
                if dec or ('not' in entry['type'] and not dec):
                    x, y, _ = recurse_decisions(decisions, entry['ifBody'],
                                                tick_location + f' ifBody:{idx}', action_hash)
                    actions += x
                    locations += y
                else:
                    x, y, _ = recurse_decisions(decisions, entry['elseBody'],
                                                tick_location + f' elseBody:{idx}', action_hash)
                    actions += x
                    locations += y
            else:
                return actions
        elif entry['type'] == 'while':
            if len(decisions) > 0:
                while True:
                    dec = decisions.pop(0)
                    if dec or ('not' in entry['type'] and not dec):
                        x, y, _ = recurse_decisions(decisions, entry['body'],
                                                    tick_location + f' body:{idx}', action_hash)
                        actions += x
                        locations += y
                    else:
                        break
                    if len(decisions) == 0:
                        return actions
            else:
                return actions
        elif entry['type'] == 'repeatUntil':
            if len(decisions) > 0:
                while True:
                    dec = decisions.pop(0)
                    x, y, _ = recurse_decisions(decisions, entry['body'],
                                                tick_location + f' body:{idx}', action_hash)
                    actions += x
                    locations += y
                    if invert_repeat_until:
                        dec = not dec
                    if dec or ('not' in entry['type'] and not dec):
                        break
                    if len(decisions) == 0:
                        return actions, locations
            else:
                return actions, locations

    return actions, locations


def get_starting_actions_from_given_decisions(decisions, code_json, action_hash,
                                              invert_repeat_until=False):
    actions = recurse_decisions(decisions, code_json['body'], action_hash, invert_repeat_until)
    return actions


def take_action(action, heroRow, heroCol, heroDir):
    newRow = heroRow
    newCol = heroCol
    newDir = heroDir
    if action == 'move':
        if (heroDir == 'north'): newRow = heroRow + 1
        if (heroDir == 'south'): newRow = heroRow - 1
        if (heroDir == 'east'): newCol = heroCol + 1
        if (heroDir == 'west'): newCol = heroCol - 1
    if action == 'turnLeft':
        if (heroDir == 'north'):
            newDir = 'west'
        elif (heroDir == 'south'):
            newDir = 'east'
        elif (heroDir == 'east'):
            newDir = 'north'
        elif (heroDir == 'west'):
            newDir = 'south'
    if action == 'turnRight':
        if (heroDir == 'north'):
            newDir = 'east'
        elif (heroDir == 'south'):
            newDir = 'west'
        elif (heroDir == 'east'):
            newDir = 'south'
        elif (heroDir == 'west'):
            newDir = 'north'

    return newRow, newCol, newDir


def simulate_actions(actions, blocked, visited, heroRow, heroCol, heroDir, markers=None):
    new_visited = copy.deepcopy(visited)
    crashed = False

    if markers is not None:
        current_markers = copy.deepcopy(markers)
        next_markers = copy.deepcopy(markers)

    for action in actions:
        newRow, newCol, newDir = take_action(action, heroRow, heroCol, heroDir)
        if newRow < 0 or newRow >= len(blocked) or newCol < 0 or newCol >= len(
                blocked[0]) or blocked[newRow][newCol]:
            crashed = True
            break
        else:
            if heroRow != newRow or heroCol != newCol:
                new_visited[newRow][newCol] += 1
            heroRow = newRow
            heroCol = newCol
            heroDir = newDir
            # new_visited[heroRow][heroCol] += 1

        if markers is not None and action == 'pickMarker':
            current_markers[heroRow][newCol] += 1
            next_markers[heroRow][newCol] = max(0, next_markers[heroRow][newCol] - 1)
        if markers is not None and action == 'putMarker':
            next_markers[heroRow][newCol] += 1

    if markers is not None:
        return crashed, new_visited, heroRow, heroCol, heroDir, current_markers, next_markers

    return crashed, new_visited, heroRow, heroCol, heroDir


def get_heuristics_array(heroRow, heroCol, heroDir, emulator, next_actions, locations=None):
    full_actions = emulator.state.actions + next_actions
    crashed, visited, newHeroRow, newHeroCol, newHeroDir = \
        simulate_actions(next_actions, emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir)

    no_moves, no_turns, no_segments, \
        no_long_segments, no_pick_markers, no_put_markers, no_turn_segments = \
        count_codeworld_quality_from_emulator_actions_and_visitation(
            full_actions,
            visited)

    n = max(emulator.state.world.rows, emulator.state.world.cols)

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

    no_turn_segments = 1 - min(no_turn_segments, 1)

    if crashed:
        alive = 0
    else:
        alive = 1

    ones = (visited == 1).sum()
    others = (visited > 1).sum()

    if ones + others == 0:
        vis_quality = 0
    else:
        vis_quality = ones / (ones + others)

    array = np.array([
        no_moves,
        no_turns,
        no_segments,
        no_long_segments,
        no_pick_markers,
        no_put_markers,
        no_turn_segments,
        alive,
        vis_quality
    ])

    if locations is not None:
        locations_set = set(locations)
        for tick in emulator.state.ticks:
            locations_set.add(str(tick.location))
        cov = len(locations_set) / emulator.ast.total_count
        # array = np.append(array, cov)

    quad = get_quadrant_from_position(newHeroRow, newHeroCol,
                                      emulator.state.world.rows,
                                      emulator.state.world.cols)

    return array, quad, newHeroDir


def get_heuristics_for_decision(decision: str, emulator: FastEmulator,
                                initial_lookahead_size: int = 0):
    if 'binary' in decision:
        next_actions, locations = get_next_actions_from_decision_position_code(
            decision,
            emulator.current_location,
            str(emulator.location),
            emulator.ast.getJson(),
            emulator.action_hash)
        heroRow = emulator.state.world.heroRow
        heroCol = emulator.state.world.heroCol
        heroDir = emulator.state.world.heroDir

        array, quad, dr = get_heuristics_array(heroRow, heroCol, heroDir, emulator,
                                               next_actions, locations=locations)

    elif 'dir' in decision:
        if '1' in decision:
            heroDir = 'north'
        elif '2' in decision:
            heroDir = 'east'
        elif '3' in decision:
            heroDir = 'south'
        elif '4' in decision:
            heroDir = 'west'
        else:
            raise Exception('Invalid direction decision')

        if initial_lookahead_size > 0:
            jsn = json.dumps(emulator.ast.getJson())
            if all(x in jsn for x in ['if', 'repeatUntil']):
                next_actions = [list(i) for i in itertools.product([False, True],
                                                                   repeat=initial_lookahead_size)]
            else:
                next_actions = [get_starting_actions_from_given_decisions(
                    [True for _ in range(initial_lookahead_size)],
                    emulator.ast.getJson(),
                    emulator.action_hash,
                    True), get_starting_actions_from_given_decisions(
                    [False for _ in range(initial_lookahead_size)],
                    emulator.ast.getJson(),
                    emulator.action_hash,
                )]
        else:
            next_actions = [get_next_actions_from_decision_position_code(
                decision,
                emulator.current_location,
                emulator.ast.getJson(),
                emulator.action_hash)]

        heroRow = emulator.state.world.heroRow
        heroCol = emulator.state.world.heroCol

        aux = []
        for next_action_lst in next_actions:
            if not heroRow or not heroCol:
                aux2 = [get_heuristics_array(y, x, heroDir, emulator,
                                             next_action_lst)
                        for y, x in [get_position_from_quadrant(quad,
                                                                emulator.state.world.rows,
                                                                emulator.state.world.cols)
                                     for quad in [Quadrant.center, Quadrant.top_left,
                                                  Quadrant.top_right,
                                                  Quadrant.bottom_left,
                                                  Quadrant.bottom_right]]]

            else:
                aux2 = [get_heuristics_array(heroRow, heroCol, heroDir, emulator,
                                             next_action_lst)]

            aux += aux2

        sm = np.sum([x[0] for x in aux], axis=1)
        idx = np.random.choice(np.flatnonzero(np.isclose(sm, sm.max())))
        array, quad, dr = aux[idx]

    elif 'pos' in decision:
        if '0' in decision:
            heroRow, heroCol = get_position_from_quadrant(Quadrant.center,
                                                          emulator.state.world.rows,
                                                          emulator.state.world.cols)
        elif '1' in decision:
            heroRow, heroCol = get_position_from_quadrant(Quadrant.bottom_left,
                                                          emulator.state.world.rows,
                                                          emulator.state.world.cols)
        elif '2' in decision:
            heroRow, heroCol = get_position_from_quadrant(Quadrant.top_left,
                                                          emulator.state.world.rows,
                                                          emulator.state.world.cols)
        elif '3' in decision:
            heroRow, heroCol = get_position_from_quadrant(Quadrant.bottom_right,
                                                          emulator.state.world.rows,
                                                          emulator.state.world.cols)
        elif '4' in decision:
            heroRow, heroCol = get_position_from_quadrant(Quadrant.top_right,
                                                          emulator.state.world.rows,
                                                          emulator.state.world.cols)

        else:
            raise Exception('Invalid position decision')

        aux = []
        if initial_lookahead_size > 0:
            jsn = json.dumps(emulator.ast.getJson())
            if all(x in jsn for x in ['if', 'repeatUntil']):
                next_actions = [list(i) for i in itertools.product([False, True],
                                                                   repeat=initial_lookahead_size)]
            else:
                next_actions = [get_starting_actions_from_given_decisions(
                    [True for _ in range(initial_lookahead_size)],
                    emulator.ast.getJson(),
                    emulator.action_hash,
                    True), get_starting_actions_from_given_decisions(
                    [False for _ in range(initial_lookahead_size)],
                    emulator.ast.getJson(),
                    emulator.action_hash,
                )]
        else:
            next_actions = [get_next_actions_from_decision_position_code(
                decision,
                emulator.current_location,
                emulator.ast.getJson(),
                emulator.action_hash)]

        for next_action_lst in next_actions:
            heroDir = emulator.state.world.heroDir
            if heroDir:
                aux2 = [get_heuristics_array(heroRow, heroCol, heroDir, emulator,
                                             next_action_lst)]
            else:
                aux2 = [get_heuristics_array(heroRow, heroCol, 'north', emulator,
                                             next_action_lst),
                        get_heuristics_array(heroRow, heroCol, 'east', emulator,
                                             next_action_lst),
                        get_heuristics_array(heroRow, heroCol, 'south', emulator,
                                             next_action_lst),
                        get_heuristics_array(heroRow, heroCol, 'west', emulator,
                                             next_action_lst)]
            aux += aux2

            sm = np.sum([x[0] for x in aux], axis=1)
            idx = np.random.choice(np.flatnonzero(np.isclose(sm, sm.max())))
            array, quad, dr = aux[idx]

    else:
        raise Exception('Invalid decision')

    if quad == Quadrant.center:
        quad = 0
    elif quad == Quadrant.bottom_left:
        quad = 1
    elif quad == Quadrant.top_left:
        quad = 2
    elif quad == Quadrant.bottom_right:
        quad = 3
    elif quad == Quadrant.top_right:
        quad = 4
    else:
        print(quad)
        raise Exception('Invalid quadrant')

    if dr == 'north':
        dr = 1
    elif dr == 'east':
        dr = 2
    elif dr == 'south':
        dr = 3
    elif dr == 'west':
        dr = 4
    else:
        print(dr)
        raise Exception('Invalid direction')

    quad = np.eye(5)[quad]
    dr = np.eye(4)[dr - 1]

    array = np.append(array, quad, axis=0)
    array = np.append(array, dr, axis=0)

    array = torch.from_numpy(array).float()

    # if array[-2] == 0:
    #     print(emulator.state.world.draw())

    return {"features": array}


def lookahead_features_pre_process_input_hoc(decision: str, emulator: FastEmulator):
    lookahead_features = get_heuristics_for_decision(decision, emulator,
                                                     initial_lookahead_size=5)
    current_features = get_location_quality_coverage_tensor(emulator)[len(
        location2idx):]

    features = torch.cat((lookahead_features['features'],
                          torch.tensor(current_features).float()), dim=0)

    return {"features": features}


def get_quality_indicators(heroRow, heroCol, heroDir, emulator, next_actions):
    full_actions = emulator.state.actions + next_actions
    crashed, visited, newHeroRow, newHeroCol, newHeroDir = \
        simulate_actions(next_actions, emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir)

    no_moves, no_turns, no_segments, \
        no_long_segments, no_pick_markers, no_put_markers, no_turn_segments = \
        count_codeworld_quality_from_emulator_actions_and_visitation(
            full_actions,
            visited)

    ones = (visited == 1).sum()
    others = (visited > 1).sum()

    if ones + others == 0:
        visited = 1
    else:
        visited = ones / (ones + others)

    return crashed, visited, no_moves, no_turns, no_segments, \
        no_long_segments, no_pick_markers, no_put_markers, no_turn_segments


def count_maximum_actions_in_body(body):
    count = 0
    for x in body:
        if x['type'] in ['move', 'turnRight', 'turnLeft', 'pickMarker', 'putMarker']:
            count += 1
        elif x['type'] == 'if':
            count += count_maximum_actions_in_body(x['body'])
        elif x['type'] == 'repeatUntil':
            count += count_maximum_actions_in_body(x['body'])
        elif x['type'] == 'ifElse':
            count += max(count_maximum_actions_in_body(x['ifBody']), count_maximum_actions_in_body(x['elseBody']))
        elif x['type'] == 'repeat':
            count += count_maximum_actions_in_body(x['body']) * x['times']
        elif x['type'] == 'while':
            count += count_maximum_actions_in_body(x['body'])

    return count


def get_count_of_actions_for_decision(decision, current_location, full_code):
    location_dict = json.loads(current_location)
    if 'loc' in decision:
        return count_maximum_actions_in_body(full_code['body'])
    elif 'binary' in decision:
        dec = ('1' in decision and 'no' not in location_dict['condition'] and location_dict[
            'type'] != 'repeatUntil') or (
                      'no' in location_dict['condition'] and '0' in decision) or (
                      location_dict['type'] == 'repeatUntil' and '0' in decision)
        if dec:
            return count_maximum_actions_in_body(full_code['body'])
        else:
            full_code = json.dumps(full_code)
            bdy = full_code.split(current_location)[1]
            idx = 0
            pair = 0
            for i, x in enumerate(bdy):
                if x == '[':
                    pair += 1
                elif x == ']':
                    pair -= 1
                if pair == -1:
                    idx = i
                    break
            bdy = bdy[:idx]
            if "," in bdy:
                bdy = bdy[2:]
            # if '[' in bdy:
            #     bdy += ']}'
            try:
                bdy = json.loads('[' + bdy + ']')
            except:
                print(bdy)
                raise Exception('Invalid body')
            try:
                q = count_maximum_actions_in_body(bdy)
                return q
            except:
                print(bdy)
                raise Exception('Invalid body')


def get_next_actions_and_locations_from_decision(decision, emulator):
    if 'binary' in decision:
        next_actions, locations = get_next_actions_from_decision_position_code(
            decision,
            emulator.current_location,
            str(emulator.location),
            emulator.ast.getJson(),
            emulator.action_hash)
        heroRow = emulator.state.world.heroRow
        heroCol = emulator.state.world.heroCol
        heroDir = emulator.state.world.heroDir
        action_count = get_count_of_actions_for_decision(decision,
                                                         emulator.current_location,
                                                         emulator.ast.getJson())

    elif 'loc' in decision:
        quad, heroDir = idx2quad_dir[int(decision.split(':')[-1])]
        heroRow, heroCol = get_position_from_quadrant(quad,
                                                      emulator.state.world.rows,
                                                      emulator.state.world.cols)
        next_actions, locations = get_next_actions_from_decision_position_code(
            decision,
            emulator.current_location,
            str(emulator.location),
            emulator.ast.getJson(),
            emulator.action_hash)
        action_count = get_count_of_actions_for_decision(decision,
                                                         emulator.current_location,
                                                         emulator.ast.getJson())

    else:
        raise Exception('Invalid decision')

    return next_actions, locations, heroRow, heroCol, heroDir, action_count


def get_metrics_for_decision_dif(decision, emulator):
    next_actions, locations, heroRow, heroCol, heroDir = \
        get_next_actions_and_locations_from_decision(decision, emulator)

    locations += [str(x.location) for x in emulator.state.ticks]

    crashed, visited, no_moves, no_turns, no_segments, \
        no_long_segments, no_pick_markers, no_put_markers, no_turn_segments = \
        get_quality_indicators(heroRow, heroCol, heroDir, emulator, next_actions)

    return crashed, visited, no_moves, no_turns, no_segments, \
        no_long_segments, no_pick_markers, no_put_markers, no_turn_segments, locations


def get_current_metrics_dif(emulator):
    heroRow = emulator.state.world.heroRow
    heroCol = emulator.state.world.heroCol
    heroDir = emulator.state.world.heroDir

    crashed, visited, no_moves, no_turns, no_segments, \
        no_long_segments, no_pick_markers, no_put_markers, no_turn_segments = \
        get_quality_indicators(heroRow, heroCol, heroDir, emulator, [])

    locations = [str(x.location) for x in emulator.state.ticks]
    if emulator.location is not None:
        locations += [str(emulator.location)]

    return crashed, visited, no_moves, no_turns, no_segments, \
        no_long_segments, no_pick_markers, no_put_markers, no_turn_segments, locations


def dif_features_pre_process_input_hoc(decision: str, emulator: FastEmulator):
    lookahead_crashed, lookahead_visited, lookahead_no_moves, \
        lookahead_no_turns, lookahead_no_segments, lookahead_no_long_segments, \
        lookahead_no_pick_markers, lookahead_no_put_markers, \
        lookahead_no_turn_segments, lookahead_locations = get_metrics_for_decision_dif(decision, emulator)
    current_crashed, current_visited, current_no_moves, \
        current_no_turns, current_no_segments, current_no_long_segments, \
        current_no_pick_markers, current_no_put_markers, \
        current_no_turn_segments, current_locations = get_current_metrics_dif(emulator)

    if lookahead_visited < 1:
        lookahead_crashed = 1

    features_list = [bool(1 - lookahead_crashed),
                     len(set(current_locations)) < len(set(lookahead_locations)),
                     current_no_long_segments < lookahead_no_long_segments,
                     current_no_segments < lookahead_no_segments,
                     current_no_moves < lookahead_no_moves,
                     current_no_turns < lookahead_no_turns,
                     ]

    curr_loc_dict = json.loads(emulator.current_location)

    if curr_loc_dict['type'] == 'repeatUntil':
        features_list += [True, False, False, False]
    elif curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse':
        if curr_loc_dict['condition'] == 'frontIsClear':
            features_list += [False, True, False, False]
        elif curr_loc_dict['condition'] == 'leftIsClear':
            features_list += [False, False, True, False]
        elif curr_loc_dict['condition'] == 'rightIsClear':
            features_list += [False, False, False, True]
        else:
            features_list += [False, False, False, False]
    else:
        features_list += [False, False, False, False]

    if 'binary:1' in decision:
        features_list += [True]
    else:
        features_list += [False]

    features = torch.tensor(features_list).float()

    return {"features": features}


def dif_features_pre_process_input_karel(decision: str, emulator: FastEmulator):
    lookahead_crashed, lookahead_visited, lookahead_no_moves, \
        lookahead_no_turns, lookahead_no_segments, lookahead_no_long_segments, \
        lookahead_no_pick_markers, lookahead_no_put_markers, \
        lookahead_no_turn_segments, lookahead_locations = get_metrics_for_decision_dif(decision, emulator)
    current_crashed, current_visited, current_no_moves, \
        current_no_turns, current_no_segments, current_no_long_segments, \
        current_no_pick_markers, current_no_put_markers, \
        current_no_turn_segments, current_locations = get_current_metrics_dif(emulator)

    if lookahead_visited < 1:
        lookahead_crashed = 1

    features_list = [bool(1 - lookahead_crashed),
                     len(set(current_locations)) < len(set(lookahead_locations)),
                     current_no_long_segments < lookahead_no_long_segments,
                     current_no_segments < lookahead_no_segments,
                     current_no_moves < lookahead_no_moves,
                     current_no_turns < lookahead_no_turns,
                     current_no_pick_markers < lookahead_no_pick_markers,
                     current_no_put_markers < lookahead_no_put_markers
                     ]

    curr_loc_dict = json.loads(emulator.current_location)

    if curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'frontIsClear':
        features_list += [True, False, False, False, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'notFrontIsClear':
        features_list += [False, True, False, False, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'leftIsClear':
        features_list += [False, False, True, False, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'rightIsClear':
        features_list += [False, False, False, True, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'markersPresent':
        features_list += [False, False, False, False, True, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'noMarkersPresent':
        features_list += [False, False, False, False, False, True, False, False, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'frontIsClear':
        features_list += [False, False, False, False, False, False, True, False, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'notFrontIsClear':
        features_list += [False, False, False, False, False, False, False, True, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'leftIsClear':
        features_list += [False, False, False, False, False, False, False, False, True, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'rightIsClear':
        features_list += [False, False, False, False, False, False, False, False, False, True, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'markersPresent':
        features_list += [False, False, False, False, False, False, False, False, False, False, True, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'noMarkersPresent':
        features_list += [False, False, False, False, False, False, False, False, False, False, False, True]
    else:
        features_list += [False, False, False, False, False, False, False, False, False, False, False, False]

    if 'binary:1' in decision:
        features_list += [True]
    else:
        features_list += [False]

    features = torch.tensor(features_list).float()

    return {"features": features}


def dif_features_pre_process_input_karel_compact(decision: str, emulator: FastEmulator):
    lookahead_crashed, lookahead_visited, lookahead_no_moves, \
        lookahead_no_turns, lookahead_no_segments, lookahead_no_long_segments, \
        lookahead_no_pick_markers, lookahead_no_put_markers, \
        lookahead_no_turn_segments, lookahead_locations = get_metrics_for_decision_dif(decision, emulator)
    current_crashed, current_visited, current_no_moves, \
        current_no_turns, current_no_segments, current_no_long_segments, \
        current_no_pick_markers, current_no_put_markers, \
        current_no_turn_segments, current_locations = get_current_metrics_dif(emulator)

    if lookahead_visited < 1:
        lookahead_crashed = 1

    features_list = [bool(1 - lookahead_crashed),
                     len(set(current_locations)) < len(set(lookahead_locations)),
                     current_no_long_segments < lookahead_no_long_segments,
                     current_no_segments < lookahead_no_segments,
                     current_no_moves < lookahead_no_moves,
                     current_no_turns < lookahead_no_turns,
                     current_no_pick_markers < lookahead_no_pick_markers,
                     current_no_put_markers < lookahead_no_put_markers
                     ]

    curr_loc_dict = json.loads(emulator.current_location)

    if curr_loc_dict['type'] == 'while':
        features_list += [True]
    else:
        features_list += [False]

    if curr_loc_dict['type'] == 'if':
        features_list += [True]
    else:
        features_list += [False]

    if 'condition' in curr_loc_dict:
        if 'no' in curr_loc_dict['condition']:
            features_list += [True]
        else:
            features_list += [False]
        if curr_loc_dict['condition'] == 'frontIsClear' or curr_loc_dict['condition'] == 'notFrontIsClear':
            features_list += [True]
        else:
            features_list += [False]
        if curr_loc_dict['condition'] == 'leftIsClear':
            features_list += [True]
        else:
            features_list += [False]
        if curr_loc_dict['condition'] == 'rightIsClear':
            features_list += [True]
        else:
            features_list += [False]
        if curr_loc_dict['condition'] == 'markersPresent' or curr_loc_dict['condition'] == 'noMarkersPresent':
            features_list += [True]
        else:
            features_list += [False]
    else:
        features_list += [False, False, False, False, False]

    if 'binary:1' in decision:
        features_list += [True]
    else:
        features_list += [False]

    features = torch.tensor(features_list).float()

    return {"features": features}


def grid_and_coverage_pre_process_input_hoc(decision: str, emulator: FastEmulator):
    next_actions, new_locations, heroRow, heroCol, heroDir, action_count = \
        get_next_actions_and_locations_from_decision(decision, emulator)
    current_locations = [str(x.location) for x in emulator.state.ticks]
    if emulator.location is not None:
        current_locations += [str(emulator.location)]
    # full_actions = emulator.state.actions + next_actions
    new_crashed, new_visited, new_hero_row, new_hero_col, new_hero_dir = \
        simulate_actions(next_actions, emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir)
    current_crashed, current_visited, current_hero_row, current_hero_col, current_hero_dir = \
        simulate_actions([], emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir)

    new_crashed = new_crashed or (new_visited > 1).any()
    new_visited = new_visited != 0
    current_visited = current_visited != 0
    new_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    new_hero_loc[new_hero_row, new_hero_col] = 1
    current_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    current_hero_loc[current_hero_row, current_hero_col] = 1
    # stack visited, location and blocked
    # how about hero direction? four channels?
    if new_crashed:
        grid = np.zeros((12, emulator.state.world.rows, emulator.state.world.cols))
    else:
        current_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        if current_hero_dir and current_hero_row and current_hero_col:
            current_hero_loc[emulator.state.world.hero_dir_to_int(current_hero_dir) - 1,
            current_hero_row, current_hero_col] \
                = 1

        new_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        new_hero_loc[emulator.state.world.hero_dir_to_int(new_hero_dir) - 1, new_hero_row, new_hero_col] \
            = 1

        grid = np.stack([new_visited, current_visited, emulator.state.world.blocked, emulator.state.world.unknown],
                        axis=0)
        grid = np.concatenate([grid, new_hero_loc, current_hero_loc], axis=0)
    grid = torch.tensor(grid).float()
    # features includes only improvement in coverage??
    curr_loc_dict = json.loads(emulator.current_location)

    features_list = [len(set(current_locations)) < len(set(current_locations + new_locations))]
    # features_list = [False]
    if curr_loc_dict['type'] == 'repeatUntil':
        features_list += [True, False, False, False]
    elif curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse':
        if curr_loc_dict['condition'] == 'frontIsClear':
            features_list += [False, True, False, False]
        elif curr_loc_dict['condition'] == 'leftIsClear':
            features_list += [False, False, True, False]
        elif curr_loc_dict['condition'] == 'rightIsClear':
            features_list += [False, False, False, True]
        else:
            features_list += [False, False, False, False]
    else:
        features_list += [False, False, False, False]

    if 'binary:1' in decision:
        features_list += [True]
    else:
        features_list += [False]

    # features_list = [False, False, False, False, False, False]

    features = torch.tensor(features_list).float()
    return {"grid": grid, "features": features}


def grid_and_coverage_action_count_pre_process_input_hoc(decision: str, emulator: FastEmulator):
    next_actions, new_locations, heroRow, heroCol, heroDir, action_count = \
        get_next_actions_and_locations_from_decision(decision, emulator)
    current_locations = [str(x.location) for x in emulator.state.ticks]
    if emulator.location is not None:
        current_locations += [str(emulator.location)]
    # full_actions = emulator.state.actions + next_actions
    new_crashed, new_visited, new_hero_row, new_hero_col, new_hero_dir = \
        simulate_actions(next_actions, emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir)
    current_crashed, current_visited, current_hero_row, current_hero_col, current_hero_dir = \
        simulate_actions([], emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir)

    new_crashed = new_crashed or (new_visited > 1).any()
    new_visited = new_visited != 0
    current_visited = current_visited != 0
    new_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    new_hero_loc[new_hero_row, new_hero_col] = 1
    current_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    current_hero_loc[current_hero_row, current_hero_col] = 1
    # stack visited, location and blocked
    # how about hero direction? four channels?
    if new_crashed:
        grid = np.zeros((12, emulator.state.world.rows, emulator.state.world.cols))
    else:
        current_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        if current_hero_dir and current_hero_row and current_hero_col:
            current_hero_loc[emulator.state.world.hero_dir_to_int(current_hero_dir) - 1,
            current_hero_row, current_hero_col] \
                = 1

        new_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        new_hero_loc[emulator.state.world.hero_dir_to_int(new_hero_dir) - 1, new_hero_row, new_hero_col] \
            = 1

        grid = np.stack([new_visited, current_visited, emulator.state.world.blocked, emulator.state.world.unknown],
                        axis=0)
        grid = np.concatenate([grid, new_hero_loc, current_hero_loc], axis=0)
    grid = torch.tensor(grid).float()
    # features includes only improvement in coverage??
    curr_loc_dict = json.loads(emulator.current_location)

    features_list = [len(set(current_locations)) < len(set(current_locations + new_locations))]
    # features_list = [False]
    if curr_loc_dict['type'] == 'repeatUntil':
        features_list += [True, False, False, False]
    elif curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse':
        if curr_loc_dict['condition'] == 'frontIsClear':
            features_list += [False, True, False, False]
        elif curr_loc_dict['condition'] == 'leftIsClear':
            features_list += [False, False, True, False]
        elif curr_loc_dict['condition'] == 'rightIsClear':
            features_list += [False, False, False, True]
        else:
            features_list += [False, False, False, False]
    else:
        features_list += [False, False, False, False]

    if 'binary:1' in decision:
        features_list += [True]
    else:
        features_list += [False]

    if action_count == 0:
        features_list += [False, False, False]
    elif action_count == 1:
        features_list += [True, False, False]
    elif action_count == 2:
        features_list += [False, True, False]
    elif action_count == 3:
        features_list += [False, False, True]
    else:
        features_list += [True, True, True]

    features = torch.tensor(features_list).float()
    return {"grid": grid, "features": features}


def next_grid_only_and_coverage_pre_process_input_hoc(decision: str, emulator: FastEmulator):
    next_actions, new_locations, heroRow, heroCol, heroDir = \
        get_next_actions_and_locations_from_decision(decision, emulator)
    current_locations = [str(x.location) for x in emulator.state.ticks]
    if emulator.location is not None:
        current_locations += [str(emulator.location)]
    # full_actions = emulator.state.actions + next_actions
    new_crashed, new_visited, new_hero_row, new_hero_col, new_hero_dir = \
        simulate_actions(next_actions, emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir)

    new_crashed = new_crashed or (new_visited > 1).any()
    new_visited = new_visited != 0
    new_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    new_hero_loc[new_hero_row, new_hero_col] = 1
    # current_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    # current_hero_loc[emulator.state.heroRow, emulator.state.heroCol] = 1
    # current_hero_dir = emulator.state.heroDir
    # stack visited, location and blocked
    # how about hero direction? four channels?
    if new_crashed:
        grid = np.zeros((7, emulator.state.world.rows, emulator.state.world.cols))
    else:
        # current_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        # if current_hero_dir and emulator.state.heroRow and emulator.state.heroCol:
        #     current_hero_loc[emulator.state.world.hero_dir_to_int(current_hero_dir) - 1,
        #     emulator.state.heroRow, emulator.state.heroCol] \
        #         = 1

        new_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        new_hero_loc[emulator.state.world.hero_dir_to_int(new_hero_dir) - 1, new_hero_row, new_hero_col] \
            = 1

        grid = np.stack([new_visited, emulator.state.world.blocked, emulator.state.world.unknown],
                        axis=0)
        grid = np.concatenate([grid, new_hero_loc], axis=0)
    grid = torch.tensor(grid).float()
    # features includes only improvement in coverage??
    curr_loc_dict = json.loads(emulator.current_location)

    features_list = [len(set(current_locations)) < len(set(current_locations + new_locations))]
    if curr_loc_dict['type'] == 'repeatUntil':
        features_list += [True, False, False, False]
    elif curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse':
        if curr_loc_dict['condition'] == 'frontIsClear':
            features_list += [False, True, False, False]
        elif curr_loc_dict['condition'] == 'leftIsClear':
            features_list += [False, False, True, False]
        elif curr_loc_dict['condition'] == 'rightIsClear':
            features_list += [False, False, False, True]
        else:
            features_list += [False, False, False, False]
    else:
        features_list += [False, False, False, False]

    if 'binary:1' in decision:
        features_list += [True]
    else:
        features_list += [False]

    features = torch.tensor(features_list).float()
    return {"grid": grid, "features": features}


def grid_and_coverage_action_count_pre_process_input_karel(decision: str, emulator: FastEmulator):
    next_actions, new_locations, heroRow, heroCol, heroDir, action_count = \
        get_next_actions_and_locations_from_decision(decision, emulator)
    current_locations = [str(x.location) for x in emulator.state.ticks]
    if emulator.location is not None:
        current_locations += [str(emulator.location)]
    # full_actions = emulator.state.actions + next_actions
    new_crashed, new_visited, new_hero_row, new_hero_col, new_hero_dir, current_markers, next_markers = \
        simulate_actions(next_actions, emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir, emulator.state.world.markers)
    current_crashed, current_visited, current_hero_row, current_hero_col, current_hero_dir = \
        simulate_actions([], emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir)

    # transform markers to one-hot
    current_markers = np.array(current_markers)
    next_markers = np.array(next_markers)

    new_current_markers = np.zeros((10, emulator.state.world.rows, emulator.state.world.cols))
    new_next_markers = np.zeros((10, emulator.state.world.rows, emulator.state.world.cols))
    for i in range(current_markers.shape[0]):
        for j in range(current_markers.shape[1]):
            if current_markers[i, j] != 0:
                new_current_markers[min(9, current_markers[i, j] - 1), i, j] = 1
    for i in range(next_markers.shape[0]):
        for j in range(next_markers.shape[1]):
            if next_markers[i, j] != 0:
                new_next_markers[min(9, current_markers[i, j] - 1), i, j] = 1

    current_markers = new_current_markers.astype(int)
    next_markers = new_next_markers.astype(int)

    new_crashed = new_crashed or (new_visited > 1).any()
    new_visited = new_visited != 0
    current_visited = current_visited != 0
    new_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    new_hero_loc[new_hero_row, new_hero_col] = 1
    current_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    current_hero_loc[current_hero_row, current_hero_col] = 1
    # stack visited, location and blocked
    # how about hero direction? four channels?
    if new_crashed:
        grid = np.zeros((32, emulator.state.world.rows, emulator.state.world.cols))
    else:
        current_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        if current_hero_dir and current_hero_row and current_hero_col:
            current_hero_loc[emulator.state.world.hero_dir_to_int(current_hero_dir) - 1,
            current_hero_row, current_hero_col] \
                = 1

        new_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        new_hero_loc[emulator.state.world.hero_dir_to_int(new_hero_dir) - 1, new_hero_row, new_hero_col] \
            = 1

        grid = np.stack([new_visited,
                         current_visited,
                         emulator.state.world.blocked,
                         emulator.state.world.unknown,
                         ],
                        axis=0)
        grid = np.concatenate([grid, next_markers, current_markers], axis=0)
        grid = np.concatenate([grid, new_hero_loc, current_hero_loc], axis=0)
    grid = torch.tensor(grid).float()
    # features includes only improvement in coverage??
    curr_loc_dict = json.loads(emulator.current_location)

    features_list = [len(set(current_locations)) < len(set(current_locations + new_locations))]
    if curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'frontIsClear':
        features_list += [True, False, False, False, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'notFrontIsClear':
        features_list += [False, True, False, False, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'leftIsClear':
        features_list += [False, False, True, False, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'rightIsClear':
        features_list += [False, False, False, True, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'markersPresent':
        features_list += [False, False, False, False, True, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'noMarkersPresent':
        features_list += [False, False, False, False, False, True, False, False, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'frontIsClear':
        features_list += [False, False, False, False, False, False, True, False, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'notFrontIsClear':
        features_list += [False, False, False, False, False, False, False, True, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'leftIsClear':
        features_list += [False, False, False, False, False, False, False, False, True, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'rightIsClear':
        features_list += [False, False, False, False, False, False, False, False, False, True, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'markersPresent':
        features_list += [False, False, False, False, False, False, False, False, False, False, True, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'noMarkersPresent':
        features_list += [False, False, False, False, False, False, False, False, False, False, False, True]
    else:
        features_list += [False, False, False, False, False, False, False, False, False, False, False, False]

    if 'binary:1' in decision:
        features_list += [True]
    else:
        features_list += [False]

    if action_count == 0:
        features_list += [False, False, False]
    elif action_count == 1:
        features_list += [True, False, False]
    elif action_count == 2:
        features_list += [False, True, False]
    elif action_count == 3:
        features_list += [False, False, True]
    else:
        features_list += [True, True, True]

    # features_list += [True, True, True]

    features = torch.tensor(features_list).float()
    return {"grid": grid, "features": features}


def grid_and_coverage_action_count_pre_process_input_karel_markerbitmap(decision: str, emulator: FastEmulator):
    next_actions, new_locations, heroRow, heroCol, heroDir, action_count = \
        get_next_actions_and_locations_from_decision(decision, emulator)
    current_locations = [str(x.location) for x in emulator.state.ticks]
    if emulator.location is not None:
        current_locations += [str(emulator.location)]
    # full_actions = emulator.state.actions + next_actions
    new_crashed, new_visited, new_hero_row, new_hero_col, new_hero_dir, current_markers, next_markers = \
        simulate_actions(next_actions, emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir, emulator.state.world.markers)
    current_crashed, current_visited, current_hero_row, current_hero_col, current_hero_dir = \
        simulate_actions([], emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir)

    current_markers = np.array(current_markers)
    current_markers = np.expand_dims(current_markers, axis=0)
    next_markers = np.array(next_markers)
    next_markers = np.expand_dims(next_markers, axis=0)
    current_markers = current_markers != 0
    next_markers = next_markers != 0

    new_crashed = new_crashed or (new_visited > 1).any()
    new_visited = new_visited != 0
    current_visited = current_visited != 0
    new_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    new_hero_loc[new_hero_row, new_hero_col] = 1
    current_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    current_hero_loc[current_hero_row, current_hero_col] = 1
    # stack visited, location and blocked
    # how about hero direction? four channels?
    if new_crashed:
        grid = np.zeros((14, emulator.state.world.rows, emulator.state.world.cols))
    else:
        current_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        if current_hero_dir and current_hero_row and current_hero_col:
            current_hero_loc[emulator.state.world.hero_dir_to_int(current_hero_dir) - 1,
            current_hero_row, current_hero_col] \
                = 1

        new_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        new_hero_loc[emulator.state.world.hero_dir_to_int(new_hero_dir) - 1, new_hero_row, new_hero_col] \
            = 1

        grid = np.stack([new_visited,
                         current_visited,
                         emulator.state.world.blocked,
                         emulator.state.world.unknown,
                         ],
                        axis=0)
        grid = np.concatenate([grid, next_markers, current_markers], axis=0)
        grid = np.concatenate([grid, new_hero_loc, current_hero_loc], axis=0)
    grid = torch.tensor(grid).float()
    # features includes only improvement in coverage??
    curr_loc_dict = json.loads(emulator.current_location)

    features_list = [len(set(current_locations)) < len(set(current_locations + new_locations))]
    if curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'frontIsClear':
        features_list += [True, False, False, False, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'notFrontIsClear':
        features_list += [False, True, False, False, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'leftIsClear':
        features_list += [False, False, True, False, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'rightIsClear':
        features_list += [False, False, False, True, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'markersPresent':
        features_list += [False, False, False, False, True, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'noMarkersPresent':
        features_list += [False, False, False, False, False, True, False, False, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'frontIsClear':
        features_list += [False, False, False, False, False, False, True, False, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'notFrontIsClear':
        features_list += [False, False, False, False, False, False, False, True, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'leftIsClear':
        features_list += [False, False, False, False, False, False, False, False, True, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'rightIsClear':
        features_list += [False, False, False, False, False, False, False, False, False, True, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'markersPresent':
        features_list += [False, False, False, False, False, False, False, False, False, False, True, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'noMarkersPresent':
        features_list += [False, False, False, False, False, False, False, False, False, False, False, True]
    else:
        features_list += [False, False, False, False, False, False, False, False, False, False, False, False]

    if 'binary:1' in decision:
        features_list += [True]
    else:
        features_list += [False]

    if action_count == 0:
        features_list += [False, False, False]
    elif action_count == 1:
        features_list += [True, False, False]
    elif action_count == 2:
        features_list += [False, True, False]
    elif action_count == 3:
        features_list += [False, False, True]
    else:
        features_list += [True, True, True]

    features = torch.tensor(features_list).float()
    return {"grid": grid, "features": features}


def grid_and_coverage_action_count_pre_process_input_compact_karel_markerbitmap(decision: str, emulator: FastEmulator):
    next_actions, new_locations, heroRow, heroCol, heroDir, action_count = \
        get_next_actions_and_locations_from_decision(decision, emulator)
    current_locations = [str(x.location) for x in emulator.state.ticks]
    if emulator.location is not None:
        current_locations += [str(emulator.location)]
    # full_actions = emulator.state.actions + next_actions
    new_crashed, new_visited, new_hero_row, new_hero_col, new_hero_dir, current_markers, next_markers = \
        simulate_actions(next_actions, emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir, emulator.state.world.markers)
    current_crashed, current_visited, current_hero_row, current_hero_col, current_hero_dir = \
        simulate_actions([], emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir)

    current_markers = np.array(current_markers)
    current_markers = np.expand_dims(current_markers, axis=0)
    next_markers = np.array(next_markers)
    next_markers = np.expand_dims(next_markers, axis=0)
    current_markers = current_markers != 0
    next_markers = next_markers != 0

    new_crashed = new_crashed or (new_visited > 1).any()
    new_visited = new_visited != 0
    current_visited = current_visited != 0
    new_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    new_hero_loc[new_hero_row, new_hero_col] = 1
    current_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    current_hero_loc[current_hero_row, current_hero_col] = 1
    # stack visited, location and blocked
    # how about hero direction? four channels?
    if new_crashed:
        grid = np.zeros((14, emulator.state.world.rows, emulator.state.world.cols))
    else:
        current_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        if current_hero_dir and current_hero_row and current_hero_col:
            current_hero_loc[emulator.state.world.hero_dir_to_int(current_hero_dir) - 1,
            current_hero_row, current_hero_col] \
                = 1

        new_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        new_hero_loc[emulator.state.world.hero_dir_to_int(new_hero_dir) - 1, new_hero_row, new_hero_col] \
            = 1

        grid = np.stack([new_visited,
                         current_visited,
                         emulator.state.world.blocked,
                         emulator.state.world.unknown,
                         ],
                        axis=0)
        grid = np.concatenate([grid, next_markers, current_markers], axis=0)
        grid = np.concatenate([grid, new_hero_loc, current_hero_loc], axis=0)
    grid = torch.tensor(grid).float()
    # features includes only improvement in coverage??
    curr_loc_dict = json.loads(emulator.current_location)

    features_list = [len(set(current_locations)) < len(set(current_locations + new_locations))]
    if curr_loc_dict['type'] == 'while' and 'markersPresent' in curr_loc_dict['condition']:
        features_list += [True, False, False, False, False, False]
    elif (curr_loc_dict['type'] == 'while' and 'frontIsClear' in curr_loc_dict['condition']):
        features_list += [False, True, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and 'markersPresent' in curr_loc_dict['condition']:
        features_list += [False, False, True, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and 'frontIsClear' in curr_loc_dict['condition']:
        features_list += [False, False, False, True, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and 'leftIsClear' in curr_loc_dict['condition']:
        features_list += [False, False, False, False, True, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and 'rightIsClear' in curr_loc_dict['condition']:
        features_list += [False, False, False, False, False, True]
    else:
        features_list += [False, False, False, False, False, False]

    if 'condition' in curr_loc_dict and 'no' in curr_loc_dict['condition']:
        features_list += [True]
    else:
        features_list += [False]

    if 'binary:1' in decision:
        features_list += [True]
    else:
        features_list += [False]

    if action_count == 0:
        features_list += [False, False, False]
    elif action_count == 1:
        features_list += [True, False, False]
    elif action_count == 2:
        features_list += [False, True, False]
    elif action_count == 3:
        features_list += [False, False, True]
    else:
        features_list += [True, True, True]

    features = torch.tensor(features_list).float()
    return {"grid": grid, "features": features}


def grid_and_coverage_pre_process_input_karel(decision: str, emulator: FastEmulator):
    next_actions, new_locations, heroRow, heroCol, heroDir = \
        get_next_actions_and_locations_from_decision(decision, emulator)
    current_locations = [str(x.location) for x in emulator.state.ticks]
    if emulator.location is not None:
        current_locations += [str(emulator.location)]
    # full_actions = emulator.state.actions + next_actions
    new_crashed, new_visited, new_hero_row, new_hero_col, new_hero_dir, current_markers, next_markers = \
        simulate_actions(next_actions, emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir, emulator.state.world.markers)
    current_crashed, current_visited, current_hero_row, current_hero_col, current_hero_dir = \
        simulate_actions([], emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir)

    # transform markers to one-hot
    current_markers = np.array(current_markers)
    next_markers = np.array(next_markers)

    new_current_markers = np.zeros((10, emulator.state.world.rows, emulator.state.world.cols))
    new_next_markers = np.zeros((10, emulator.state.world.rows, emulator.state.world.cols))
    for i in range(current_markers.shape[0]):
        for j in range(current_markers.shape[1]):
            if current_markers[i, j] != 0:
                new_current_markers[min(9, current_markers[i, j] - 1), i, j] = 1
    for i in range(next_markers.shape[0]):
        for j in range(next_markers.shape[1]):
            if next_markers[i, j] != 0:
                new_next_markers[min(9, current_markers[i, j] - 1), i, j] = 1

    current_markers = new_current_markers.astype(int)
    next_markers = new_next_markers.astype(int)

    new_crashed = new_crashed or (new_visited > 1).any()
    new_visited = new_visited != 0
    current_visited = current_visited != 0
    new_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    new_hero_loc[new_hero_row, new_hero_col] = 1
    current_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    current_hero_loc[current_hero_row, current_hero_col] = 1
    # stack visited, location and blocked
    # how about hero direction? four channels?
    if new_crashed:
        grid = np.zeros((32, emulator.state.world.rows, emulator.state.world.cols))
    else:
        current_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        if current_hero_dir and current_hero_row and current_hero_col:
            current_hero_loc[emulator.state.world.hero_dir_to_int(current_hero_dir) - 1,
            current_hero_row, current_hero_col] \
                = 1

        new_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        new_hero_loc[emulator.state.world.hero_dir_to_int(new_hero_dir) - 1, new_hero_row, new_hero_col] \
            = 1

        grid = np.stack([new_visited,
                         current_visited,
                         emulator.state.world.blocked,
                         emulator.state.world.unknown,
                         ],
                        axis=0)
        grid = np.concatenate([grid, next_markers, current_markers], axis=0)
        grid = np.concatenate([grid, new_hero_loc, current_hero_loc], axis=0)
    grid = torch.tensor(grid).float()
    # features includes only improvement in coverage??
    curr_loc_dict = json.loads(emulator.current_location)

    features_list = [len(set(current_locations)) < len(set(current_locations + new_locations))]
    if curr_loc_dict['type'] == 'while':
        features_list += [True]
    else:
        features_list += [False]

    if curr_loc_dict['type'] == 'if':
        features_list += [True]
    else:
        features_list += [False]

    if 'condition' in curr_loc_dict:
        if 'no' in curr_loc_dict['condition']:
            features_list += [True]
        else:
            features_list += [False]
        if curr_loc_dict['condition'] == 'frontIsClear' or curr_loc_dict['condition'] == 'notFrontIsClear':
            features_list += [True]
        else:
            features_list += [False]
        if curr_loc_dict['condition'] == 'leftIsClear':
            features_list += [True]
        else:
            features_list += [False]
        if curr_loc_dict['condition'] == 'rightIsClear':
            features_list += [True]
        else:
            features_list += [False]
        if curr_loc_dict['condition'] == 'markersPresent' or curr_loc_dict['condition'] == 'noMarkersPresent':
            features_list += [True]
        else:
            features_list += [False]
    else:
        features_list += [False, False, False, False, False]

    if 'binary:1' in decision:
        features_list += [True]
    else:
        features_list += [False]

    features = torch.tensor(features_list).float()
    return {"grid": grid, "features": features}


def grid_and_coverage_pre_process_input_karel_extended(decision: str, emulator: FastEmulator):
    next_actions, new_locations, heroRow, heroCol, heroDir = \
        get_next_actions_and_locations_from_decision(decision, emulator)
    current_locations = [str(x.location) for x in emulator.state.ticks]
    if emulator.location is not None:
        current_locations += [str(emulator.location)]
    # full_actions = emulator.state.actions + next_actions
    new_crashed, new_visited, new_hero_row, new_hero_col, new_hero_dir, current_markers, next_markers = \
        simulate_actions(next_actions, emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir, emulator.state.world.markers)
    current_crashed, current_visited, current_hero_row, current_hero_col, current_hero_dir = \
        simulate_actions([], emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir)

    # transform markers to one-hot
    current_markers = np.array(current_markers)
    next_markers = np.array(next_markers)

    new_current_markers = np.zeros((10, emulator.state.world.rows, emulator.state.world.cols))
    new_next_markers = np.zeros((10, emulator.state.world.rows, emulator.state.world.cols))
    for i in range(current_markers.shape[0]):
        for j in range(current_markers.shape[1]):
            if current_markers[i, j] != 0:
                new_current_markers[min(9, current_markers[i, j] - 1), i, j] = 1
    for i in range(next_markers.shape[0]):
        for j in range(next_markers.shape[1]):
            if next_markers[i, j] != 0:
                new_next_markers[min(9, current_markers[i, j] - 1), i, j] = 1

    current_markers = new_current_markers.astype(int)
    next_markers = new_next_markers.astype(int)

    new_crashed = new_crashed or (new_visited > 1).any()
    new_visited = new_visited != 0
    current_visited = current_visited != 0
    new_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    new_hero_loc[new_hero_row, new_hero_col] = 1
    current_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    current_hero_loc[current_hero_row, current_hero_col] = 1
    # stack visited, location and blocked
    # how about hero direction? four channels?
    if new_crashed:
        grid = np.zeros((32, emulator.state.world.rows, emulator.state.world.cols))
    else:
        current_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        if current_hero_dir and current_hero_row and current_hero_col:
            current_hero_loc[emulator.state.world.hero_dir_to_int(current_hero_dir) - 1,
            current_hero_row, current_hero_col] \
                = 1

        new_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        new_hero_loc[emulator.state.world.hero_dir_to_int(new_hero_dir) - 1, new_hero_row, new_hero_col] \
            = 1

        grid = np.stack([new_visited,
                         current_visited,
                         emulator.state.world.blocked,
                         emulator.state.world.unknown,
                         ],
                        axis=0)
        grid = np.concatenate([grid, next_markers, current_markers], axis=0)
        grid = np.concatenate([grid, new_hero_loc, current_hero_loc], axis=0)
    grid = torch.tensor(grid).float()
    # features includes only improvement in coverage??
    curr_loc_dict = json.loads(emulator.current_location)

    features_list = [len(set(current_locations)) < len(set(current_locations + new_locations))]
    if curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'frontIsClear':
        features_list += [True, False, False, False, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'notFrontIsClear':
        features_list += [False, True, False, False, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'leftIsClear':
        features_list += [False, False, True, False, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'rightIsClear':
        features_list += [False, False, False, True, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'markersPresent':
        features_list += [False, False, False, False, True, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'noMarkersPresent':
        features_list += [False, False, False, False, False, True, False, False, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'frontIsClear':
        features_list += [False, False, False, False, False, False, True, False, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'notFrontIsClear':
        features_list += [False, False, False, False, False, False, False, True, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'leftIsClear':
        features_list += [False, False, False, False, False, False, False, False, True, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'rightIsClear':
        features_list += [False, False, False, False, False, False, False, False, False, True, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'markersPresent':
        features_list += [False, False, False, False, False, False, False, False, False, False, True, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'noMarkersPresent':
        features_list += [False, False, False, False, False, False, False, False, False, False, False, True]
    else:
        features_list += [False, False, False, False, False, False, False, False, False, False, False, False]

    if 'binary:1' in decision:
        features_list += [True]
    else:
        features_list += [False]

    features = torch.tensor(features_list).float()
    return {"grid": grid, "features": features}


def grid_and_coverage_pre_process_input_karel_for_filtered(decision: str, emulator: FastEmulator):
    next_actions, new_locations, heroRow, heroCol, heroDir = \
        get_next_actions_and_locations_from_decision(decision, emulator)
    current_locations = [str(x.location) for x in emulator.state.ticks]
    if emulator.location is not None:
        current_locations += [str(emulator.location)]
    # full_actions = emulator.state.actions + next_actions
    new_crashed, new_visited, new_hero_row, new_hero_col, new_hero_dir, current_markers, next_markers = \
        simulate_actions(next_actions, emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir, emulator.state.world.markers)
    current_crashed, current_visited, current_hero_row, current_hero_col, current_hero_dir = \
        simulate_actions([], emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir)

    # transform markers to one-hot
    current_markers = np.array(current_markers)
    next_markers = np.array(next_markers)

    new_current_markers = np.zeros((10, emulator.state.world.rows, emulator.state.world.cols))
    new_next_markers = np.zeros((10, emulator.state.world.rows, emulator.state.world.cols))
    for i in range(current_markers.shape[0]):
        for j in range(current_markers.shape[1]):
            if current_markers[i, j] != 0:
                new_current_markers[min(9, current_markers[i, j] - 1), i, j] = 1
    for i in range(next_markers.shape[0]):
        for j in range(next_markers.shape[1]):
            if next_markers[i, j] != 0:
                new_next_markers[min(9, current_markers[i, j] - 1), i, j] = 1

    current_markers = new_current_markers.astype(int)
    next_markers = new_next_markers.astype(int)

    new_crashed = new_crashed or (new_visited > 1).any()
    new_visited = new_visited != 0
    current_visited = current_visited != 0
    new_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    new_hero_loc[new_hero_row, new_hero_col] = 1
    current_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    current_hero_loc[current_hero_row, current_hero_col] = 1
    # stack visited, location and blocked
    # how about hero direction? four channels?
    if new_crashed:
        grid = np.zeros((32, emulator.state.world.rows, emulator.state.world.cols))
    else:
        current_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        if current_hero_dir and current_hero_row and current_hero_col:
            current_hero_loc[emulator.state.world.hero_dir_to_int(current_hero_dir) - 1,
            current_hero_row, current_hero_col] \
                = 1

        new_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        new_hero_loc[emulator.state.world.hero_dir_to_int(new_hero_dir) - 1, new_hero_row, new_hero_col] \
            = 1

        grid = np.stack([new_visited,
                         current_visited,
                         emulator.state.world.blocked,
                         emulator.state.world.unknown,
                         ],
                        axis=0)
        grid = np.concatenate([grid, next_markers, current_markers], axis=0)
        grid = np.concatenate([grid, new_hero_loc, current_hero_loc], axis=0)
    grid = torch.tensor(grid).float()
    # features includes only improvement in coverage??
    curr_loc_dict = json.loads(emulator.current_location)

    features_list = [len(set(current_locations)) < len(set(current_locations + new_locations))]
    if curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'frontIsClear':
        features_list += [True, False, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'notFrontIsClear':
        features_list += [False, True, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'markersPresent':
        features_list += [False, False, True, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'noMarkersPresent':
        features_list += [False, False, False, True, False, False, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'frontIsClear':
        features_list += [False, False, False, False, True, False, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'notFrontIsClear':
        features_list += [False, False, False, False, False, True, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'leftIsClear':
        features_list += [False, False, False, False, False, False, True, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'rightIsClear':
        features_list += [False, False, False, False, False, False, False, True, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'markersPresent':
        features_list += [False, False, False, False, False, False, False, False, True, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'noMarkersPresent':
        features_list += [False, False, False, False, False, False, False, False, False, True]
    else:
        features_list += [False, False, False, False, False, False, False, False, False, False]

    if 'binary:1' in decision:
        features_list += [True]
    else:
        features_list += [False]

    features = torch.tensor(features_list).float()
    return {"grid": grid, "features": features}


def grid_and_coverage_pre_process_input_karel_binary_markers_for_filtered(decision: str, emulator: FastEmulator):
    next_actions, new_locations, heroRow, heroCol, heroDir = \
        get_next_actions_and_locations_from_decision(decision, emulator)
    current_locations = [str(x.location) for x in emulator.state.ticks]
    if emulator.location is not None:
        current_locations += [str(emulator.location)]
    # full_actions = emulator.state.actions + next_actions
    new_crashed, new_visited, new_hero_row, new_hero_col, new_hero_dir, current_markers, next_markers = \
        simulate_actions(next_actions, emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir, emulator.state.world.markers)
    current_crashed, current_visited, current_hero_row, current_hero_col, current_hero_dir = \
        simulate_actions([], emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir)

    current_markers = np.array(current_markers)
    current_markers = np.expand_dims(current_markers, axis=0)
    next_markers = np.array(next_markers)
    next_markers = np.expand_dims(next_markers, axis=0)
    current_markers = current_markers != 0
    next_markers = next_markers != 0

    new_crashed = new_crashed or (new_visited > 1).any()
    new_visited = new_visited != 0
    current_visited = current_visited != 0
    new_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    new_hero_loc[new_hero_row, new_hero_col] = 1
    current_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    current_hero_loc[current_hero_row, current_hero_col] = 1
    # stack visited, location and blocked
    # how about hero direction? four channels?
    if new_crashed:
        grid = np.zeros((14, emulator.state.world.rows, emulator.state.world.cols))
    else:
        current_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        if current_hero_dir and current_hero_row and current_hero_col:
            current_hero_loc[emulator.state.world.hero_dir_to_int(current_hero_dir) - 1,
            current_hero_row, current_hero_col] \
                = 1

        new_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        new_hero_loc[emulator.state.world.hero_dir_to_int(new_hero_dir) - 1, new_hero_row, new_hero_col] \
            = 1

        grid = np.stack([new_visited,
                         current_visited,
                         emulator.state.world.blocked,
                         emulator.state.world.unknown,
                         ],
                        axis=0)
        grid = np.concatenate([grid, next_markers, current_markers], axis=0)
        grid = np.concatenate([grid, new_hero_loc, current_hero_loc], axis=0)
    grid = torch.tensor(grid).float()
    # features includes only improvement in coverage??
    curr_loc_dict = json.loads(emulator.current_location)

    features_list = [len(set(current_locations)) < len(set(current_locations + new_locations))]
    if curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'frontIsClear':
        features_list += [True, False, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'notFrontIsClear':
        features_list += [False, True, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'markersPresent':
        features_list += [False, False, True, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'noMarkersPresent':
        features_list += [False, False, False, True, False, False, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'frontIsClear':
        features_list += [False, False, False, False, True, False, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'notFrontIsClear':
        features_list += [False, False, False, False, False, True, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'leftIsClear':
        features_list += [False, False, False, False, False, False, True, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'rightIsClear':
        features_list += [False, False, False, False, False, False, False, True, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'markersPresent':
        features_list += [False, False, False, False, False, False, False, False, True, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'noMarkersPresent':
        features_list += [False, False, False, False, False, False, False, False, False, True]
    else:
        features_list += [False, False, False, False, False, False, False, False, False, False]

    if 'binary:1' in decision:
        features_list += [True]
    else:
        features_list += [False]

    features = torch.tensor(features_list).float()
    return {"grid": grid, "features": features}


def grid_and_coverage_pre_process_input_karel_nomarkernumber(decision: str, emulator: FastEmulator):
    next_actions, new_locations, heroRow, heroCol, heroDir = \
        get_next_actions_and_locations_from_decision(decision, emulator)
    current_locations = [str(x.location) for x in emulator.state.ticks]
    if emulator.location is not None:
        current_locations += [str(emulator.location)]
    # full_actions = emulator.state.actions + next_actions
    new_crashed, new_visited, new_hero_row, new_hero_col, new_hero_dir, current_markers, next_markers = \
        simulate_actions(next_actions, emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir, emulator.state.world.markers)
    current_crashed, current_visited, current_hero_row, current_hero_col, current_hero_dir = \
        simulate_actions([], emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir)

    # transform markers to one-hot
    current_markers = np.array(current_markers)
    current_markers = np.expand_dims(current_markers, axis=0)
    next_markers = np.array(next_markers)
    next_markers = np.expand_dims(next_markers, axis=0)
    current_markers = current_markers != 0
    next_markers = next_markers != 0

    new_crashed = new_crashed or (new_visited > 1).any()
    new_visited = new_visited != 0
    current_visited = current_visited != 0
    new_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    new_hero_loc[new_hero_row, new_hero_col] = 1
    current_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    current_hero_loc[current_hero_row, current_hero_col] = 1
    # stack visited, location and blocked
    # how about hero direction? four channels?
    if new_crashed:
        grid = np.zeros((12, emulator.state.world.rows, emulator.state.world.cols))
    else:
        current_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        if current_hero_dir and current_hero_row and current_hero_col:
            current_hero_loc[emulator.state.world.hero_dir_to_int(current_hero_dir) - 1,
            current_hero_row, current_hero_col] \
                = 1

        new_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        new_hero_loc[emulator.state.world.hero_dir_to_int(new_hero_dir) - 1, new_hero_row, new_hero_col] \
            = 1

        grid = np.stack([new_visited,
                         current_visited,
                         emulator.state.world.blocked,
                         emulator.state.world.unknown,
                         ],
                        axis=0)
        # grid = np.concatenate([grid, next_markers, current_markers], axis=0)
        grid = np.concatenate([grid, new_hero_loc, current_hero_loc], axis=0)
    grid = torch.tensor(grid).float()
    # features includes only improvement in coverage??
    curr_loc_dict = json.loads(emulator.current_location)

    features_list = [len(set(current_locations)) < len(set(current_locations + new_locations))]
    if curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'frontIsClear':
        features_list += [True, False, False, False, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'notFrontIsClear':
        features_list += [False, True, False, False, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'leftIsClear':
        features_list += [False, False, True, False, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'rightIsClear':
        features_list += [False, False, False, True, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'markersPresent':
        features_list += [False, False, False, False, True, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'noMarkersPresent':
        features_list += [False, False, False, False, False, True, False, False, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'frontIsClear':
        features_list += [False, False, False, False, False, False, True, False, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'notFrontIsClear':
        features_list += [False, False, False, False, False, False, False, True, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'leftIsClear':
        features_list += [False, False, False, False, False, False, False, False, True, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'rightIsClear':
        features_list += [False, False, False, False, False, False, False, False, False, True, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'markersPresent':
        features_list += [False, False, False, False, False, False, False, False, False, False, True, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'noMarkersPresent':
        features_list += [False, False, False, False, False, False, False, False, False, False, False, True]
    else:
        features_list += [False, False, False, False, False, False, False, False, False, False, False, False]

    if 'binary:1' in decision:
        features_list += [True]
    else:
        features_list += [False]

    if 'putMarker' in next_actions:
        features_list += [True]
    else:
        features_list += [False]

    features = torch.tensor(features_list).float()
    return {"grid": grid, "features": features}


def grid_and_coverage_pre_process_input_karel_markerbitmap(decision: str, emulator: FastEmulator):
    next_actions, new_locations, heroRow, heroCol, heroDir = \
        get_next_actions_and_locations_from_decision(decision, emulator)
    current_locations = [str(x.location) for x in emulator.state.ticks]
    if emulator.location is not None:
        current_locations += [str(emulator.location)]
    # full_actions = emulator.state.actions + next_actions
    new_crashed, new_visited, new_hero_row, new_hero_col, new_hero_dir, current_markers, next_markers = \
        simulate_actions(next_actions, emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir, emulator.state.world.markers)
    current_crashed, current_visited, current_hero_row, current_hero_col, current_hero_dir = \
        simulate_actions([], emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir)

    # transform markers to one-hot
    current_markers = np.array(current_markers)
    current_markers = np.expand_dims(current_markers, axis=0)
    next_markers = np.array(next_markers)
    next_markers = np.expand_dims(next_markers, axis=0)
    current_markers = current_markers != 0
    next_markers = next_markers != 0

    new_crashed = new_crashed or (new_visited > 1).any()
    new_visited = new_visited != 0
    current_visited = current_visited != 0
    new_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    new_hero_loc[new_hero_row, new_hero_col] = 1
    current_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    current_hero_loc[current_hero_row, current_hero_col] = 1
    # stack visited, location and blocked
    # how about hero direction? four channels?
    if new_crashed:
        grid = np.zeros((14, emulator.state.world.rows, emulator.state.world.cols))
    else:
        current_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        if current_hero_dir and current_hero_row and current_hero_col:
            current_hero_loc[emulator.state.world.hero_dir_to_int(current_hero_dir) - 1,
            current_hero_row, current_hero_col] \
                = 1

        new_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        new_hero_loc[emulator.state.world.hero_dir_to_int(new_hero_dir) - 1, new_hero_row, new_hero_col] \
            = 1

        grid = np.stack([new_visited,
                         current_visited,
                         emulator.state.world.blocked,
                         emulator.state.world.unknown,
                         ],
                        axis=0)
        grid = np.concatenate([grid, next_markers, current_markers], axis=0)
        grid = np.concatenate([grid, new_hero_loc, current_hero_loc], axis=0)
    grid = torch.tensor(grid).float()
    # features includes only improvement in coverage??
    curr_loc_dict = json.loads(emulator.current_location)

    features_list = [len(set(current_locations)) < len(set(current_locations + new_locations))]
    if curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'frontIsClear':
        features_list += [True, False, False, False, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'notFrontIsClear':
        features_list += [False, True, False, False, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'leftIsClear':
        features_list += [False, False, True, False, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'rightIsClear':
        features_list += [False, False, False, True, False, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'markersPresent':
        features_list += [False, False, False, False, True, False, False, False, False, False, False, False]
    elif curr_loc_dict['type'] == 'while' and curr_loc_dict['condition'] == 'noMarkersPresent':
        features_list += [False, False, False, False, False, True, False, False, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'frontIsClear':
        features_list += [False, False, False, False, False, False, True, False, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'notFrontIsClear':
        features_list += [False, False, False, False, False, False, False, True, False, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'leftIsClear':
        features_list += [False, False, False, False, False, False, False, False, True, False, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'rightIsClear':
        features_list += [False, False, False, False, False, False, False, False, False, True, False, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'markersPresent':
        features_list += [False, False, False, False, False, False, False, False, False, False, True, False]
    elif (curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse') \
            and curr_loc_dict['condition'] == 'noMarkersPresent':
        features_list += [False, False, False, False, False, False, False, False, False, False, False, True]
    else:
        features_list += [False, False, False, False, False, False, False, False, False, False, False, False]

    if 'binary:1' in decision:
        features_list += [True]
    else:
        features_list += [False]

    features = torch.tensor(features_list).float()
    return {"grid": grid, "features": features}


def grid_coverage_code_type_pre_process_input_hoc(decision: str, emulator: FastEmulator, code_type: int):
    assert code_type in [0, 1, 2]
    next_actions, new_locations, heroRow, heroCol, heroDir = \
        get_next_actions_and_locations_from_decision(decision, emulator)
    current_locations = [str(x.location) for x in emulator.state.ticks]
    if emulator.location is not None:
        current_locations += [str(emulator.location)]
    # full_actions = emulator.state.actions + next_actions
    new_crashed, new_visited, new_hero_row, new_hero_col, new_hero_dir = \
        simulate_actions(next_actions, emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir)
    current_crashed, current_visited, current_hero_row, current_hero_col, current_hero_dir = \
        simulate_actions([], emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir)

    new_crashed = new_crashed or (new_visited > 1).any()
    new_visited = new_visited != 0
    current_visited = current_visited != 0
    new_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    new_hero_loc[new_hero_row, new_hero_col] = 1
    current_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    current_hero_loc[current_hero_row, current_hero_col] = 1
    # stack visited, location and blocked
    # how about hero direction? four channels?
    if new_crashed:
        grid = np.zeros((12, emulator.state.world.rows, emulator.state.world.cols))
    else:
        current_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        if current_hero_dir and current_hero_row and current_hero_col:
            current_hero_loc[emulator.state.world.hero_dir_to_int(current_hero_dir) - 1,
            current_hero_row, current_hero_col] \
                = 1

        new_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        new_hero_loc[emulator.state.world.hero_dir_to_int(new_hero_dir) - 1, new_hero_row, new_hero_col] \
            = 1

        grid = np.stack([new_visited, current_visited, emulator.state.world.blocked, emulator.state.world.unknown],
                        axis=0)
        grid = np.concatenate([grid, new_hero_loc, current_hero_loc], axis=0)
    grid = torch.tensor(grid).float()
    # features includes only improvement in coverage??
    curr_loc_dict = json.loads(emulator.current_location)

    features_list = [len(set(current_locations)) < len(set(current_locations + new_locations))]
    if curr_loc_dict['type'] == 'repeatUntil':
        features_list += [True, False, False, False]
    elif curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse':
        if curr_loc_dict['condition'] == 'frontIsClear':
            features_list += [False, True, False, False]
        elif curr_loc_dict['condition'] == 'leftIsClear':
            features_list += [False, False, True, False]
        elif curr_loc_dict['condition'] == 'rightIsClear':
            features_list += [False, False, False, True]
        else:
            features_list += [False, False, False, False]
    else:
        features_list += [False, False, False, False]

    if 'binary:1' in decision:
        features_list += [True]
    else:
        features_list += [False]

    if code_type == 0:
        features_list += [True, False, False]
    elif code_type == 1:
        features_list += [False, True, False]
    elif code_type == 2:
        features_list += [False, False, True]

    features = torch.tensor(features_list).float()
    return {"grid": grid, "features": features}


def grid_coverage_latent_pre_process_input_hoc(decision: str, emulator: FastEmulator, code_type: Tensor):
    next_actions, new_locations, heroRow, heroCol, heroDir = \
        get_next_actions_and_locations_from_decision(decision, emulator)
    current_locations = [str(x.location) for x in emulator.state.ticks]
    if emulator.location is not None:
        current_locations += [str(emulator.location)]
    new_crashed, new_visited, new_hero_row, new_hero_col, new_hero_dir = \
        simulate_actions(next_actions, emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir)
    current_crashed, current_visited, current_hero_row, current_hero_col, current_hero_dir = \
        simulate_actions([], emulator.state.world.blocked,
                         emulator.visited,
                         heroRow,
                         heroCol,
                         heroDir)

    new_crashed = new_crashed or (new_visited > 1).any()
    new_visited = new_visited != 0
    current_visited = current_visited != 0
    new_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    new_hero_loc[new_hero_row, new_hero_col] = 1
    current_hero_loc = np.zeros((emulator.state.world.rows, emulator.state.world.cols))
    current_hero_loc[current_hero_row, current_hero_col] = 1
    if new_crashed:
        grid = np.zeros((12, emulator.state.world.rows, emulator.state.world.cols))
    else:
        current_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        if current_hero_dir and current_hero_row and current_hero_col:
            current_hero_loc[emulator.state.world.hero_dir_to_int(current_hero_dir) - 1,
            current_hero_row, current_hero_col] \
                = 1

        new_hero_loc = np.zeros((4, emulator.state.world.rows, emulator.state.world.cols))
        new_hero_loc[emulator.state.world.hero_dir_to_int(new_hero_dir) - 1, new_hero_row, new_hero_col] \
            = 1

        grid = np.stack([new_visited, current_visited, emulator.state.world.blocked, emulator.state.world.unknown],
                        axis=0)
        grid = np.concatenate([grid, new_hero_loc, current_hero_loc], axis=0)
    grid = torch.tensor(grid).float()
    curr_loc_dict = json.loads(emulator.current_location)

    features_list = [len(set(current_locations)) < len(set(current_locations + new_locations))]
    if curr_loc_dict['type'] == 'repeatUntil':
        features_list += [True, False, False, False]
    elif curr_loc_dict['type'] == 'if' or curr_loc_dict['type'] == 'ifElse':
        if curr_loc_dict['condition'] == 'frontIsClear':
            features_list += [False, True, False, False]
        elif curr_loc_dict['condition'] == 'leftIsClear':
            features_list += [False, False, True, False]
        elif curr_loc_dict['condition'] == 'rightIsClear':
            features_list += [False, False, False, True]
        else:
            features_list += [False, False, False, False]
    else:
        features_list += [False, False, False, False]

    if 'binary:1' in decision:
        features_list += [True]
    else:
        features_list += [False]

    features = torch.tensor(features_list).float()
    return {"grid": grid, "features": features, "latent": code_type}


def get_features_size(features_type: str):
    if features_type == "features_only_pre_process_input_karel":
        return 4 + 2 + 7 + len(location2idx)
    elif features_type == "features_only_pre_process_input_hoc":
        return 2 + 2 + 5 + len(location2idx)
    elif features_type == "features_and_symworld_pre_process_input":
        return 2 + 7 + len(location2idx)
    elif features_type == "lookahead_features_pre_process_input_hoc":
        return 2 + 5 + 9 + 9
    elif features_type == "dif_features_pre_process_input_hoc":
        return 11
    elif features_type == "dif_features_pre_process_input_karel":
        return 21
    elif features_type == "dif_features_pre_process_input_karel_compact":
        return 16
    elif features_type == "grid_and_coverage_pre_process_input_hoc":
        return {"grid": (12, 16, 16), "features": 6}
    elif features_type == "grid_and_coverage_action_count_pre_process_input_hoc":
        return {"grid": (12, 16, 16), "features": 9}
    elif features_type == "next_grid_only_and_coverage_pre_process_input_hoc":
        return {"grid": (7, 16, 16), "features": 6}
    elif features_type == "grid_and_coverage_pre_process_input_karel":
        return {"grid": (32, 16, 16), "features": 9}
    elif features_type == "grid_and_coverage_pre_process_input_karel_nomarkernumber":
        return {"grid": (12, 16, 16), "features": 15}
    elif features_type == "grid_and_coverage_pre_process_input_karel_markerbitmap":
        return {"grid": (14, 16, 16), "features": 14}
    elif features_type == "grid_and_coverage_pre_process_input_karel_extended":
        return {"grid": (32, 16, 16), "features": 14}
    elif features_type == "grid_and_coverage_pre_process_input_karel_for_filtered":
        return {"grid": (32, 16, 16), "features": 12}
    elif features_type == "grid_and_coverage_pre_process_input_karel_binary_markers_for_filtered":
        return {"grid": (14, 16, 16), "features": 12}
    elif features_type == "grid_coverage_code_type_pre_process_input_hoc":
        return {"grid": (12, 16, 16), "features": 9}
    elif features_type == "grid_coverage_latent_pre_process_input_hoc":
        return {"grid": (12, 16, 16), "features": 6, "latent": 64}
    elif features_type == "grid_and_coverage_action_count_pre_process_input_karel":
        return {"grid": (32, 16, 16), "features": 17}
    elif features_type == "grid_and_coverage_action_count_pre_process_input_karel_markerbitmap":
        return {"grid": (14, 16, 16), "features": 17}
    elif features_type == "grid_and_coverage_action_count_pre_process_input_compact_karel_markerbitmap":
        return {"grid": (14, 16, 16), "features": 12}

    else:
        raise ValueError("Unknown features type")


def get_output_size():
    return len(decision2idx)