import argparse
import json
import os
import random
from multiprocessing import Manager, Process

import numpy as np
import torch
from tqdm import tqdm

from src.codetask_scoring.finalscore import compute_evaluation_score
from src.emulator.code import Code
from src.emulator.fast_emulator import FastEmulator
from src.emulator.task import Task
from src.symexecution.decision_makers import RandomDecisionMaker

from src.symexecution.post_processor import BlockedPostProcessor, EmptySpacePostProcessor
from src.symexecution.symworld import SymWorld
from src.taskgen.decision_makers import GridActorCriticLookaheadDecisionMaker
from src.taskgen.training import MAX_WORLD_SIZE


def run_chunk(chunk, start_index, task_decision_maker, common_q, done_variable,
              taskgen_trials=100, aggregation=100, type_='hoc'):
    os.nice(15)
    for i, (code, oracle_score, sk_nb) in enumerate(chunk):
        code = Code(type_, code)
        for x in range(aggregation):
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

            common_q.put({'line_nb': start_index + i,
                          'oracle_score': oracle_score,
                          'taskgen_scores_noncumulative': scores,
                          'code': code.get_writeable_json(),
                          'sketch_nb': sk_nb,
                          'run_nb': x,
                          'type': type_})

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
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--domain', type=str, default='hoc')
    arg_parser.add_argument('--task_trials', type=int, default=100)
    arg_parser.add_argument('--decision_maker_path', type=str, default=None)
    arg_parser.add_argument('--nb_processes', type=int, default=10)
    # choose between base or neural
    arg_parser.add_argument('--algo_type', type=str, default='neural', choices=['base', 'neural'])
    arg_parser.add_argument('--seed', type=int, default=0)

    args = arg_parser.parse_args()

    type_ = args.domain
    taskgen_trials = args.task_trials
    task_decision_maker_path = args.decision_maker_path
    algo = args.algo_type
    nb_processes = args.nb_processes

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    if algo == 'base':
        task_decision_maker = RandomDecisionMaker.auto_init()
    elif algo == 'neural':
        if task_decision_maker_path == None or type_ not in task_decision_maker_path:
            if task_decision_maker_path != None:
                print(f'WARNING: task decision maker path is not {type_}, using default')
            if type_ == 'hoc':
                task_decision_maker_path = 'models/hoc/taskgen/pretrained/score_80'
            elif type_ == 'karel':
                task_decision_maker_path = 'models/karel/taskgen/pretrained/score_90'
            else:
                raise ValueError(f'Unknown type {type_}')
        task_decision_maker = GridActorCriticLookaheadDecisionMaker.load(task_decision_maker_path)
    else:
        raise ValueError(f"Unknown algo {algo}")

    aggregation = 1

    write_path = f'results/{type_}/seed_{args.seed}/Task_Abl_{task_decision_maker.__class__.__name__}.json'
    if not os.path.exists(os.path.dirname(write_path)):
        os.makedirs(os.path.dirname(write_path), exist_ok=True)

    data_path = f'data/synthetic/{type_}/test.json'
    data = []
    for line in open(data_path):
        dct = json.loads(line)
        code = dct['program_json']
        oracle_score = dct['score']
        sk_nb = dct['sketch_nb']
        data.append((code, oracle_score, sk_nb))

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
        processes.append(Process(target=run_chunk, args=(chunks[i], i * chunk_size,
                                                         task_decision_maker, common_q, done_tracker,
                                                         taskgen_trials, aggregation, type_)))
        processes[-1].start()

    for process in processes:
        process.join()

    master_proc.join()
