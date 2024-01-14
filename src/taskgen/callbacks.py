import os
import random
from multiprocessing import Manager, Process

from src.taskgen.decision_makers import GridActorCriticLookaheadDecisionMaker
from tqdm import tqdm


from src.emulator.code import Code

from src.taskgen.data import CodeDataset
from src.taskgen.decision_makers import IntelligentDecisionMaker
from src.taskgen.task_synthesizer import obtain_hoc_saturation_score_for_code_multiple_runs, \
    obtain_karel_saturation_score_for_code_multiple_runs


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


class Callback:
    def execute(self, **kwargs):
        raise NotImplementedError()


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
        score = kwargs.get('val_score', None)
        if score is None:
            return
        if self.last_score is None:
            self.last_score = score
        else:
            self.last_score = self.last_score * (1 - self.alpha) + score * self.alpha

        if self.last_score > self.best_score:
            self.best_score = self.last_score
            self.agent.eval()
            self.agent.save(f"{self.save_path}/score_{int(self.last_score * 100)}/")
            self.agent.train()

        return [(self.last_score, 'saving_score')]


def get_task_synth_succ_ratio(dct, index, no_rollouts, runs, result_dict, decision_maker, type_="hoc"):
    os.nice(15)
    code = Code.parse_json(dct)

    successes = []
    for x in range(runs):
        if type_ == "hoc":
            scores = obtain_hoc_saturation_score_for_code_multiple_runs(code, [no_rollouts], decision_maker)
        elif type_ == "karel":
            scores = obtain_karel_saturation_score_for_code_multiple_runs(code, [no_rollouts], decision_maker)
        else:
            raise NotImplementedError()

        for roll in scores:
            if scores[roll] >= 0.90 * dct['score'][0] and scores[roll] > 0:
                successes.append(1)
            else:
                successes.append(0)

    result_dict[index] = sum(successes) / len(successes)


class TaskSynthSuccRatioCallback(Callback):
    def __init__(self,
                 every_n_epochs: int,
                 agent: IntelligentDecisionMaker,
                 val_dataset: CodeDataset,
                 no_rollouts: int,
                 no_runs: int,
                 sample_size: int,
                 type_: str = "hoc",
                 prefix: str = None):
        self.agent = agent
        self.val_dataset = val_dataset
        self.every_n_epochs = every_n_epochs

        self.no_rollouts = no_rollouts
        self.no_runs = no_runs
        self.sample_size = sample_size

        self.type_ = type_
        self.prefix = prefix

    def execute(self, **kwargs):
        if kwargs['epoch'] % self.every_n_epochs != 0:
            return

        self.agent.eval()
        self.agent.has_buffer = False

        manager = Manager()
        result_dict = manager.dict()

        processes = []
        # select samples
        sample = iter_sample_fast(self.val_dataset, min(self.sample_size, len(self.val_dataset)))
        for idx, (example, code_type, _) in enumerate(sample):
            p = Process(target=get_task_synth_succ_ratio,
                        args=(example, idx, self.no_rollouts, self.no_runs, result_dict, self.agent, self.type_))
            p.start()
            processes.append(p)

        for p in tqdm(processes):
            p.join()

        self.agent.has_buffer = True
        self.agent.train()

        return [(sum(result_dict.values()) / len(result_dict),
                 'val_score' if self.prefix is None else f"{self.prefix}_val_score")]
