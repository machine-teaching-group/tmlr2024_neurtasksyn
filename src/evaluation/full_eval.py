import argparse
import copy
import glob
import json
import os
import random
from multiprocessing import Manager, Process

import numpy as np
import torch
from tqdm import tqdm

from src.codegen.callbacks import code_task_oracle
from src.codetask_scoring.finalscore import compute_evaluation_score
from src.emulator.code import Code
from src.emulator.fast_emulator import FastEmulator
from src.emulator.task import Task
from src.evaluation.utils import rollouts_lookup
from src.symexecution.decision_makers import RandomDecisionMaker
from src.symexecution.post_processor import BlockedPostProcessor, EmptySpacePostProcessor
from src.symexecution.symworld import SymWorld
from src.taskgen.decision_makers import GridActorCriticLookaheadDecisionMaker
from src.taskgen.training import MAX_WORLD_SIZE
from src.codegen.code_sketches import CODE_SKETCHES, KAREL_SKETCHES
from src.codegen.codegen import get_partial_code_from_sketch_code_comb, \
    get_control_structures_combinations, compute_available_blocks
from src.codegen.codegen import sketch2code
from src.codegen.decision_makers import LSTMDecisionMaker, RandomCodeDecisionMaker
from src.codegen.symast import SymAst, symast_to_json


def run_chunk(chunk, start_index, code_decision_maker, task_decision_maker, common_q, done_variable,
              code_trials=5, taskgen_trials=10, aggregation=100, type_='hoc'):
    os.nice(15)
    for i, (partial_code, sk_nb, max_blocks_allowed) in enumerate(chunk):
        lst = []
        for x in range(aggregation):
            for _ in range(code_trials):
                code, _ = sketch2code(copy.deepcopy(partial_code), code_decision_maker, max_blocks_allowed, type_=type_)
                code = Code.parse_json({'program_json': code, 'program_type': type_})

                oracle_trials = rollouts_lookup[type_][sk_nb]['rollouts']
                _, oracle_score = code_task_oracle(code.astJson, oracle_trials, type_=type_)
                scores = []
                for _ in range(taskgen_trials):
                    emulator = FastEmulator(1000, 1000)

                    if isinstance(task_decision_maker, GridActorCriticLookaheadDecisionMaker):
                        task_decision_maker.set_emulator(emulator)
                        code = task_decision_maker.process_code(code)
                        code_type = None
                        task_decision_maker.set_code_type(code_type)

                    ref_task = Task([], [], code.type)

                    rows, cols = MAX_WORLD_SIZE

                    symworld = SymWorld.empty_init(rows, cols, task_decision_maker)

                    res = emulator.emulate(code, symworld)

                    if code.type == "hoc":
                        post_processor = BlockedPostProcessor()
                    elif code.type == "karel":
                        post_processor = EmptySpacePostProcessor()
                    else:
                        raise ValueError(f"Unknown type {code.type}")

                    inp_world, out_world = post_processor.symworld_to_world(
                        res.outgrid)

                    task = Task([inp_world],
                                [out_world],
                                type_=code.type)

                    score, info = compute_evaluation_score(code,
                                                           task,
                                                           ref_task,
                                                           compute_visitation_quality=True,
                                                           compute_shortest_path_quality=True if code.type == "karel" else False,
                                                           ignore_diversity=True,
                                                           ignore_dissimilarity=True,
                                                           segments_weight=None
                                                           )

                    if info['redundancy'] == 'NOT RESPECTED':
                        score = 0.00
                    if info['shortest_path'] == 'NOT RESPECTED':
                        score = 0.00
                    if info['solvability'] == 'NOT RESPECTED':
                        score = 0.00
                    if info['coverage'] != 1.0:
                        score = 0.00
                    if 'vis_score' in info and info['vis_score'] != 1.0:
                        score = 0.00

                    scores.append(score)

                lst.append({'oracle_score': oracle_score,
                            'taskgen_scores_noncumulative': scores,
                            'code': code.get_writeable_json()})

            common_q.put({'line_nb': start_index + i,
                          'sketch_nb': sk_nb,
                          'run_nb': x,
                          'type': type_,
                          'allowed_blocks': max_blocks_allowed,
                          'code_struct': partial_code.to_json(),
                          'codegen_results': lst})

    done_variable.value += 1


def master_proc_func(common_q, done_variable, total_done, write_path, data_size, agg=100):
    os.nice(15)
    pbar = tqdm(total=data_size*agg)
    with open(write_path, 'w+') as f:
        while done_variable.value < total_done:
            if not common_q.empty():
                item = common_q.get()
                f.write(json.dumps(item) + '\n')
                f.flush()
                pbar.update(1)

    print("Done with master")


if __name__ == '__main__':
    print("WARNING: experiment with large waiting time")
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--domain', type=str, default='hoc', choices=['hoc', 'karel'])
    arg_parser.add_argument('--code_trials', type=int, default=5)
    arg_parser.add_argument('--task_trials', type=int, default=100)
    arg_parser.add_argument('--task_decision_maker_path', type=str, default=None)
    arg_parser.add_argument('--code_decision_maker_path', type=str, default=None)
    arg_parser.add_argument('--seed', type=int, default=0)
    arg_parser.add_argument('--nb_processes', type=int, default=10)
    # choose between base or neural
    arg_parser.add_argument('--algo_type', type=str, default='neural', choices=['base', 'neural'])

    args = arg_parser.parse_args()


    task_decision_maker_path = args.task_decision_maker_path
    code_decision_maker_path = args.code_decision_maker_path

    code_trials = args.code_trials
    taskgen_trials = args.task_trials

    seed = args.seed
    type_ = args.domain
    algo_type = args.algo_type

    nb_processes = args.nb_processes

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if type_ == 'hoc':
        if task_decision_maker_path == None or 'hoc' not in task_decision_maker_path:
            if task_decision_maker_path != None:
                print('WARNING: task decision maker path is not hoc, using default')
            task_decision_maker_path = 'models/hoc/taskgen/pretrained/score_80'
        if code_decision_maker_path == None or 'hoc' not in code_decision_maker_path:
            if code_decision_maker_path != None:
                print('WARNING: code decision maker path is not hoc, using default')
            code_decision_maker_path = 'models/hoc/codegen/pretrained/score_88'
    elif type_ == 'karel':
        if task_decision_maker_path == None or 'karel' not in task_decision_maker_path:
            if task_decision_maker_path != None:
                print('WARNING: task decision maker path is not karel, using default')
            task_decision_maker_path = 'models/karel/taskgen/pretrained/score_90'
        if code_decision_maker_path == None or 'karel' not in code_decision_maker_path:
            if code_decision_maker_path != None:
                print('WARNING: code decision maker path is not karel, using default')
            code_decision_maker_path = 'models/karel/codegen/pretrained/score_93'
    else:
        raise ValueError(f"Unknown type {type_}")

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

    aggregation = 1
    max_blocks_allowed = 16
    min_blocks_allowed = 9

    write_path = f'results/{type_}/seed_{args.seed}/Full_{code_decision_maker.__class__.__name__}_' \
                 f'{task_decision_maker.__class__.__name__}.json'
    if not os.path.exists(os.path.dirname(write_path)):
        os.makedirs(os.path.dirname(write_path), exist_ok=True)

    data_path = f'data/synthetic/{type_}/test.json'
    data = []
    for line in open(data_path):
        dct = json.loads(line)
        number = dct['sketch_nb']
        if type_ == 'hoc':
            code_type = CODE_SKETCHES[number][0]
        else:
            code_type = KAREL_SKETCHES[number][0]
        code_type = SymAst.from_json(code_type)
        lst = get_control_structures_combinations(code_type)
        # get random combination
        idx = random.randint(0, len(lst) - 1)
        x = get_partial_code_from_sketch_code_comb(copy.deepcopy(code_type), dct['program_json'],
                                                   lst[idx])
        code_type = SymAst.from_json(symast_to_json(x))

        num_blocks_allowed = compute_available_blocks(code_type, max_blocks_allowed)
        num_blocks_allowed = random.randint(max(min_blocks_allowed, 9), num_blocks_allowed)

        data.append((code_type, number, num_blocks_allowed))

    chunk_size = len(data) // nb_processes
    chunks = []
    for i in range(nb_processes):
        if i == nb_processes - 1:
            chunks.append(data[i * chunk_size:])
        else:
            chunks.append(data[i * chunk_size: (i + 1) * chunk_size])

    manager = Manager()
    common_q = manager.Queue()
    done_tracker = manager.Value('i', 0)

    master_proc = Process(target=master_proc_func, args=(common_q, done_tracker, nb_processes, write_path, len(data), aggregation))
    master_proc.start()

    processes = []
    for i in range(nb_processes):
        processes.append(Process(target=run_chunk, args=(chunks[i], i * chunk_size, code_decision_maker,
                                                         task_decision_maker, common_q, done_tracker,
                                                         code_trials, taskgen_trials,
                                                         aggregation, type_)))
        processes[-1].start()

    for process in processes:
        process.join()

    master_proc.join()
