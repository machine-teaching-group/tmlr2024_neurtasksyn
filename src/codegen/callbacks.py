import copy
import json
import multiprocessing
import os
import random

import torch
from tqdm import tqdm

from src.emulator.code import Code
# from src.codegen.callbacks import iter_sample_fast, execute_code_oracle, Callback
from src.codegen.code_sketches import CODE_SKETCHES, KAREL_SKETCHES
from src.codegen.codegen import get_control_structures_combinations, \
    get_partial_code_from_sketch_code_comb, sketch2code
from src.codegen.symast import SymAst, symast_to_json

from src.symexecution.decision_makers import RandomDecisionMaker
from src.taskgen.task_synthesizer import obtain_hoc_saturation_score_for_code, obtain_karel_saturation_score_for_code


def iter_sample_fast(iterable, samplesize):
    results = []
    iterator = iter(iterable)
    # Fill in the first samplesize elements:
    try:
        for _ in range(samplesize):
            results.append(iterator.__next__())
    except StopIteration:
        raise ValueError("Sample larger than population.")
    random.shuffle(results)  # Randomize their positions
    for i, v in enumerate(iterator, samplesize):
        r = random.randint(0, i)
        if r < samplesize:
            results[r] = v  # at a decreasing rate, replace random items
    return results


def code_task_oracle(code, rollouts, type_='hoc'):
    decision_maker2 = RandomDecisionMaker.auto_init()
    decision_maker2.eval()
    if 'body' not in code:
        print('wtf')
    code = Code.parse_json({'program_json': code, 'program_type': type_})
    if type_ == 'hoc':
        scores, task = obtain_hoc_saturation_score_for_code(code, decision_maker2, rollouts)
    else:
        scores, task = obtain_karel_saturation_score_for_code(code, decision_maker2, rollouts)
    if scores[0] > 0.0:
        return True, scores[0]
    else:
        return False, 0.0


def execute_code_oracle(i, j, codes, task_rollouts, result_dict):
    os.nice(15)
    gd, score = code_task_oracle(codes[i][j], task_rollouts)
    result_dict[i][j] = (gd, score)


class Callback:
    def execute(self, **kwargs):
        raise NotImplementedError()


class CodeSuccessValCallback(Callback):
    def __init__(self,
                 every_n_epochs: int,
                 decision_maker,
                 val_dataset,
                 code_rollouts=5,
                 task_rollouts=100_000,
                 sample_size=10,
                 type_='hoc'):
        self.decision_maker = decision_maker
        self.val_dataset = val_dataset
        self.every_n_epochs = every_n_epochs
        self.code_rollouts = code_rollouts
        self.task_rollouts = task_rollouts
        self.sample_size = sample_size
        self.type_ = type_

    def execute(self, **kwargs):
        if kwargs['epoch'] % self.every_n_epochs != 0:
            return None

        self.decision_maker.eval()

        goods = []
        codes = dict([(i, dict([(j, None) for j in range(self.code_rollouts)])) for i in range(self.sample_size)])
        with torch.no_grad():
            for i, (_, number, _, code, latent) in enumerate(iter_sample_fast(self.val_dataset, self.sample_size)):
                if self.type_ == 'hoc':
                    code_type = CODE_SKETCHES[number][0]
                else:
                    code_type = KAREL_SKETCHES[number][0]
                code_type = SymAst.from_json(code_type)
                lst = get_control_structures_combinations(code_type)
                # get random combination
                idx = random.randint(0, len(lst) - 1)
                x = get_partial_code_from_sketch_code_comb(copy.deepcopy(code_type), json.loads(code)['program_json'],
                                                           lst[idx])
                code_type = SymAst.from_json(symast_to_json(x))

                code = Code(self.type_, json.loads(code)['program_json'])
                blocks = code.total_count
                # get random number between blocks and 16
                blocks = random.randint(blocks, max(blocks + 1, 16))

                for j in range(self.code_rollouts):
                    code, _ = sketch2code(copy.deepcopy(code_type), self.decision_maker, blocks, latent=latent,
                                          type_=self.type_)
                    codes[i][j] = code

        # we initialize the shared dictionary with the sub dictionaries
        manager = multiprocessing.Manager()
        return_dict = manager.dict([(i, manager.dict([(j, None) for j in range(self.code_rollouts)])) for i in
                                    range(self.sample_size)])

        jobs = []
        for i in range(self.sample_size):
            for j in range(self.code_rollouts):
                p = multiprocessing.Process(target=execute_code_oracle,
                                            args=(i, j, codes, self.task_rollouts, return_dict))
                jobs.append(p)
                p.start()

        for proc in tqdm(jobs):
            proc.join()

        for i in range(self.sample_size):
            if any([return_dict[i][j][0] for j in range(self.code_rollouts)]):
                goods.append(True)
            else:
                goods.append(False)

        self.decision_maker.train()
        return [(sum(goods) / len(goods), f'val_task_succ_{self.code_rollouts}')]


class SavingCallback(Callback):
    def __init__(self,
                 agent,
                 save_path: str,
                 alpha=0.2):
        self.agent = agent
        self.save_path = save_path
        self.best_score = 0
        self.alpha = alpha
        self.last_score = None

    def execute(self, **kwargs):
        score = None
        for k, v in kwargs.items():
            if k.startswith('val_task_succ'):
                score = v
                break
        if score is None:
            return
        if self.last_score is None:
            self.last_score = score
        else:
            self.last_score = self.last_score * (1 - self.alpha) + score * self.alpha

        if self.last_score > self.best_score:
            self.best_score = self.last_score
            self.agent.save(f"{self.save_path}/score_{int(self.last_score * 100)}/")

        return [(self.last_score, 'val_task_succ_smooth')]
