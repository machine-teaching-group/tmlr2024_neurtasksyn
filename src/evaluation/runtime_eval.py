import argparse
import copy
import json
import os
import time
from pprint import pprint

from tqdm import tqdm

from src.codetask_scoring.finalscore import compute_evaluation_score
from src.symexecution.decision_makers import RandomDecisionMaker
from src.taskgen.decision_makers import GridActorCriticLookaheadDecisionMaker
from src.symexecution.symworld import SymWorld
from src.emulator.code import Code
from src.emulator.fast_emulator import FastEmulator
from src.symexecution.post_processor import EmptySpacePostProcessor, BlockedPostProcessor
from src.emulator.task import Task


def measure_one(type_,
                task_decision_maker,
                code,
                score_th,
                task_trials=2000000,
                time_limit=60 * 60 * 10):
    total_rollouts = 0

    init_world = SymWorld.empty_init(16, 16, None)

    if type_ == 'karel':
        post_processor = EmptySpacePostProcessor()
    else:
        post_processor = BlockedPostProcessor()

    best_score = -1
    best_task = None
    start = time.time()
    code_obj = Code.parse_json(code)

    for j in range(task_trials):
        emulator = FastEmulator(max_ticks=1000, max_actions=1000)

        if isinstance(task_decision_maker, GridActorCriticLookaheadDecisionMaker):
            task_decision_maker.set_emulator(emulator)
        current_init_world = copy.deepcopy(init_world)
        current_init_world.set_decision_maker(task_decision_maker)

        res = emulator.emulate(code_obj, current_init_world)

        inp_world, out_world = post_processor.symworld_to_world(res.outgrid)
        task = Task([inp_world],
                    [out_world],
                    type_=code_obj.type)
        score, evaluation = \
            compute_evaluation_score(code_obj,
                                     task,
                                     Task([], [], type_=code_obj.type),
                                     ignore_diversity=True,
                                     ignore_dissimilarity=True,
                                     compute_visitation_quality=True,
                                     compute_shortest_path_quality=False if code_obj.type == 'hoc' else True)

        total_rollouts += 1

        if any([evaluation['redundancy'] == 'NOT RESPECTED',
                evaluation['solvability'] == 'NOT RESPECTED',
                evaluation['shortest_path'] == 'NOT RESPECTED',
                evaluation['coverage'] != 1.0,
                (evaluation['vis_score'] != 1.0 and code_obj.type == 'hoc')
                ]
               ):
            score = 0.0

        if best_score < score:
            best_score = score
            best_task = task

        if best_score >= score_th:
            break

        if time.time() - start > time_limit:
            break

    return_dict = {'best_score': best_score,
                   'best_task': best_task.to_json(),
                   'total_rollouts': total_rollouts,
                   'time': time.time() - start,
                   'success': best_score >= score_th}

    return return_dict


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--decision_maker_path', type=str, default=None)
    arg_parser.add_argument('--percentage', type=int, default=99)
    # choose between base or neural
    arg_parser.add_argument('--algo_type', type=str, default='neural', choices=['base', 'neural'])

    easy = [0, 1, 2, 5, 6, 7]
    hard = [3, 4, 8, 9]
    ignored = [4, 7, 8, 9]

    args = arg_parser.parse_args()

    task_decision_maker_path_orig = args.decision_maker_path
    algo = args.algo_type
    percentage = args.percentage

    write_path = f'results/time_measurements_{algo}_quality_{percentage}.json'
    if not os.path.exists(os.path.dirname(write_path)):
        os.makedirs(os.path.dirname(write_path), exist_ok=True)

    if algo == 'base':
        task_decision_maker = RandomDecisionMaker.auto_init()
        print("WARNING: experiment with large waiting time")

    data_path = f'data/real-world/expert_codes.json'

    codes = []
    with open(data_path, 'r') as f:
        for line in f:
            codes.append(json.loads(line))

    bar = tqdm(total=(len(easy) + len(hard) - len(ignored)) * 3)

    write_list = []
    for i, code in enumerate(codes):
        if i in ignored:
            continue

        type_ = code['program_type']

        if algo == 'base':
            task_decision_maker = RandomDecisionMaker.auto_init()
        elif algo == 'neural':
            if task_decision_maker_path_orig == None or type_ not in task_decision_maker_path_orig:
                if task_decision_maker_path_orig != None:
                    print(f'WARNING: task decision maker path is not {type_}, using default')
                if type_ == 'hoc':
                    task_decision_maker_path = 'models/hoc/taskgen/pretrained/score_80'
                    tmp = 0.4
                elif type_ == 'karel':
                    task_decision_maker_path = 'models/karel/taskgen/pretrained/score_90'
                    tmp = 0.4
                else:
                    raise ValueError(f'Unknown type {type_}')
            task_decision_maker = GridActorCriticLookaheadDecisionMaker.load(task_decision_maker_path)
            task_decision_maker.temperature = tmp
        else:
            raise ValueError(f"Unknown algo {algo}")

        trials_list = []
        for trial in range(3):
            trials_list.append(measure_one(type_, task_decision_maker, code, code['expert_score'] * percentage / 100))
            bar.update(1)

        write_list.append({'code': code['program_json'],
                           'type': type_,
                           'score': code['expert_score'],
                           'dff': 'easy' if i in easy else 'hard',
                           'trials': trials_list})

    with open(write_path, 'w') as f:
        json.dump(write_list, f, indent=4)
