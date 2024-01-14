import argparse
import copy
import os
from pprint import pprint

from tqdm import tqdm

from src.codegen.codegen import sketch2code
from src.codegen.decision_makers import LSTMDecisionMaker, RandomCodeDecisionMaker
from src.codetask_scoring.finalscore import compute_evaluation_score
from src.emulator.code import Code
from src.emulator.fast_emulator import FastEmulator
from src.emulator.task import Task
from src.inference.utils import sk_dict
from src.symexecution.decision_makers import RandomDecisionMaker
from src.symexecution.post_processor import EmptySpacePostProcessor, BlockedPostProcessor
from src.symexecution.symworld import SymWorld
from src.taskgen.decision_makers import GridActorCriticLookaheadDecisionMaker


def get_best_code_task_pair(type_, code_decision_maker, task_decision_maker, sketch_dict, code_trials=10,
                            task_trials=100, init_world=SymWorld.empty_init(16, 16, None), maximum_blocks=9):
    if type_ == 'hoc':
        post_processor = BlockedPostProcessor()
    elif type_ == 'karel':
        post_processor = EmptySpacePostProcessor()

    best_code = None
    best_score = -1
    best_task = None
    for i in range(code_trials):
        code, _ = sketch2code(copy.deepcopy(sketch_dict), code_decision_maker, maximum_blocks, type_=type_)
        code_obj = Code.parse_json({'program_json': code, 'program_type': type_})
        # pprint(code)

        best_task_score = 0.0
        best_task_task = None
        for j in tqdm(range(task_trials)):
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

            if any([evaluation['redundancy'] == 'NOT RESPECTED',
                    evaluation['solvability'] == 'NOT RESPECTED',
                    evaluation['shortest_path'] == 'NOT RESPECTED',
                    evaluation['coverage'] != 1.0,
                    evaluation['vis_score'] != 1.0
                    ]
                   ):
                score = 0.0

            if best_score < score:
                best_score = score
                best_code = code
                best_task = task
                best_eval = evaluation

            if best_task_score < score:
                best_task_score = score
                best_task_task = task

    return best_score, best_code, best_task


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--code_trials', type=int, default=5)
    arg_parser.add_argument('--task_trials', type=int, default=100)
    arg_parser.add_argument('--maximum_blocks', type=int, default=9)
    arg_parser.add_argument('--task_decision_maker_path', type=str, default=None)
    arg_parser.add_argument('--code_decision_maker_path', type=str, default=None)
    arg_parser.add_argument('--spec_nb', type=int, default=8, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # choose between base or neural
    arg_parser.add_argument('--algo_type', type=str, default='neural', choices=['base', 'neural'])

    args = arg_parser.parse_args()


    task_decision_maker_path = args.task_decision_maker_path
    code_decision_maker_path = args.code_decision_maker_path

    spec_nb = args.spec_nb

    if spec_nb < 5:
        type_ = 'hoc'
        if task_decision_maker_path == None or 'hoc' not in task_decision_maker_path:
            if task_decision_maker_path != None:
                print('WARNING: task decision maker path is not hoc, using default')
            task_decision_maker_path = 'models/hoc/taskgen/pretrained/score_80'
        if code_decision_maker_path == None or 'hoc' not in code_decision_maker_path:
            if code_decision_maker_path != None:
                print('WARNING: code decision maker path is not hoc, using default')
            code_decision_maker_path = 'models/hoc/codegen/pretrained/score_88'
    else:
        type_ = 'karel'
        if task_decision_maker_path == None or 'karel' not in task_decision_maker_path:
            if task_decision_maker_path != None:
                print('WARNING: task decision maker path is not karel, using default')
            task_decision_maker_path = 'models/karel/taskgen/pretrained/score_90'
        if code_decision_maker_path == None or 'karel' not in code_decision_maker_path:
            if code_decision_maker_path != None:
                print('WARNING: code decision maker path is not karel, using default')
            code_decision_maker_path = 'models/karel/codegen/pretrained/score_93'

    if not os.path.exists(task_decision_maker_path):
            raise ValueError('Task decision maker path does not exist')
    if not os.path.exists(code_decision_maker_path):
        raise ValueError('Code decision maker path does not exist')

    if args.algo_type == 'neural':
        code_decision_maker = LSTMDecisionMaker.load(code_decision_maker_path)
        task_decision_maker = GridActorCriticLookaheadDecisionMaker.load(task_decision_maker_path)
    elif args.algo_type == 'base':
        code_decision_maker = RandomCodeDecisionMaker()
        task_decision_maker = RandomDecisionMaker.auto_init()
    else:
        raise ValueError('Invalid algo type')

    sketch = sk_dict[type_][f"spec{spec_nb}"]

    best_score, best_code, best_task = get_best_code_task_pair(type_, code_decision_maker, task_decision_maker, sketch,
                                                               code_trials=args.code_trials,
                                                               task_trials=args.task_trials,
                                                               maximum_blocks=args.maximum_blocks)

    print('Best score: {}'.format(best_score)
          if best_score != 0.0 else 'Failed to find a solution to satisfy all constraints. This was the best found:')
    pprint(best_code)
    print(best_task.pregrids[0].draw())
    print(best_task.postgrids[0].draw())
