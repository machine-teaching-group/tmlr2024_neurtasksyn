import copy
import heapq
import logging
import time
from statistics import mean
from typing import Optional, Tuple, List, Generator, Union, Any

# import wandb

from src.codetask_scoring.finalscore import compute_evaluation_score, \
    compute_synthesis_score_faster
from src.emulator.code import Code
from src.emulator.fast_emulator import FastEmulator
from src.emulator.task import Task
from src.emulator.tokens import blocktypes
from src.symexecution.decision_makers import DecisionMaker, \
    RandomDecisionMaker
from src.symexecution.post_processor import PostProcessor, BlockedPostProcessor
from src.symexecution.symworld import SymWorld
from src.taskgen.decision_makers import IntelligentDecisionMaker

DEBUG = False
TIME_DEBUG = False

MAX_GRID_SIZE = (16, 16)

POOL_SIZE = 1_000_000

def avg(lst: Generator) -> float:
    lst = list(lst)
    if len(lst) == 0:
        return 0.0
    return mean(lst)


class TaskSynthesizer:
    def __init__(self,
                 code: Code,
                 ref_task: Optional[Task],
                 decision_maker: DecisionMaker,
                 post_processor: PostProcessor,
                 max_grid_size: Optional[Tuple[int, int]] = None,
                 max_iterations: int = 1000000,
                 patience: int = None,
                 diff_threshold: float = None):
        self.code = code
        self.ref_task = ref_task
        self.max_iterations = max_iterations
        self.decision_maker = decision_maker
        self.post_processor = post_processor

        if max_grid_size is None:
            self.max_grid_size = MAX_GRID_SIZE
        else:
            self.max_grid_size = max_grid_size

        self.init_task = Task([], [], code.type)

        self.patience = patience
        self.diff_threshold = diff_threshold

    def synthesize(self, num_tasks_to_generate,
                   num_blocks_allowed: int,
                   type_blocks_allowed: str,
                   type_: str,
                   log_freq: Optional[int] = 1,
                   init_symworlds: Optional[List[SymWorld]] = None,
                   ignore_diversity: bool = False,
                   ignore_dissimilarity: bool = False,
                   visitation_quality: bool = False,
                   compute_shortest_path_quality: bool = False,
                   ) -> tuple[Task, list[Any], list[Any]]:
        assert type_ in ["karel", "hoc"]

        scores = []
        heaps = []
        if init_symworlds is None:
            init_symworlds = []

        for i in range(num_tasks_to_generate):
            if i < len(init_symworlds):
                inp_world, out_world, score, heap = self.synthesize_world(
                    init_world=init_symworlds[i],
                    log_freq=log_freq,
                    log_prefix=f"world_{i}",
                    ignore_diversity=ignore_diversity,
                    ignore_dissimilarity=ignore_dissimilarity,
                    visitation_quality=visitation_quality,
                    compute_shortest_path_quality=compute_shortest_path_quality,
                )
            else:
                inp_world, out_world, score, heap = self.synthesize_world(
                    log_freq=log_freq,
                    log_prefix=f"world_{i}",
                    ignore_diversity=ignore_diversity,
                    ignore_dissimilarity=ignore_dissimilarity,
                    visitation_quality=visitation_quality,
                    compute_shortest_path_quality=compute_shortest_path_quality,
                )

            self.init_task = Task(self.init_task.pregrids + [inp_world],
                                  self.init_task.postgrids + [out_world],
                                  type_=type_,
                                  num_blocks_allowed=num_blocks_allowed,
                                  type_blocks_allowed=type_blocks_allowed,
                                  num_examples=len(
                                      self.init_task.pregrids) + 1)
            scores.append(score)
            heaps.append(heap)

        return self.init_task, scores, heaps

    def synthesize_world(self,
                         init_world: SymWorld = None,
                         log_freq: Optional[int] = 1,
                         ignore_dissimilarity: bool = False,
                         ignore_diversity: bool = False,
                         visitation_quality: bool = False,
                         compute_shortest_path_quality: bool = False,
                         log_prefix: str = None) -> Union[
        tuple[Optional[Any], Optional[Any], float, None], tuple[Any, Any, float, list[Any]]]:

        # print("synthesizing world")

        assert self.patience is None or self.patience > 0
        assert self.diff_threshold is None or self.diff_threshold > 0
        assert self.patience is None and self.diff_threshold is None or \
               self.patience is not None and self.diff_threshold is not None
        emulator = FastEmulator(max_ticks=1000, max_actions=1000)

        if isinstance(self.decision_maker, IntelligentDecisionMaker):
            self.decision_maker.set_emulator(emulator)

        max_score = -1.0
        best_inp_world = None
        best_out_world = None
        best_info = None

        heap = []

        if log_prefix:
            log_prefix = f"{log_prefix}/"
        else:
            log_prefix = ""

        if TIME_DEBUG:
            time_log = []
            time_from_worlds = []
            time_world_conv = []
            time_score = []
            time_emulation = []
            time_solv = []
            time_qual = []
            time_div = []
            time_diss = []
            time_cov = []
            time_score_exec = []

        time_lst = []
        patience = self.patience
        for i in range(self.max_iterations):
            start_all = time.time()
            if init_world is None:
                current_init_world = \
                    SymWorld.empty_init(self.max_grid_size[0],
                                        self.max_grid_size[1],
                                        decision_maker=self.decision_maker)
            else:
                current_init_world = copy.deepcopy(init_world)
                current_init_world.set_decision_maker(self.decision_maker)

            if TIME_DEBUG:
                start = time.time()

            result = emulator.emulate(self.code, current_init_world)
            # print('emulated')
            if TIME_DEBUG:
                end = time.time()
                time_emulation.append(end - start)

            # IMPORTANT: due to deep copies, the original decision maker is not
            # affected by the emulation process, so it will take the same decisions
            # next time. Thus, we pass the decision maker which was actually used
            # back to the self.decision_maker
            self.decision_maker = result.outgrid.decision_maker
            if isinstance(self.decision_maker, IntelligentDecisionMaker):
                self.decision_maker.reset_buffer()

            if TIME_DEBUG:
                start = time.time()
            inp_world, out_world = self.post_processor.symworld_to_world(result.outgrid)
            if self.code.type == "hoc":
                out_world.heroDir = "any"
            if TIME_DEBUG:
                end = time.time()
                time_world_conv.append(end - start)

            if TIME_DEBUG:
                start = time.time()
            task = Task(self.init_task.pregrids + [inp_world],
                        self.init_task.postgrids + [out_world],
                        type_=self.code.type)
            if TIME_DEBUG:
                end = time.time()
                time_from_worlds.append(end - start)

            if TIME_DEBUG:
                start = time.time()
                score, info, solv_time, qual_time, diss_time, div_time, cov_time, \
                    exec_time = compute_synthesis_score_faster(result, self.code,
                                                               task, self.ref_task,
                                                               compute_visitation_quality=visitation_quality, )
                time_solv.append(solv_time)
                time_qual.append(qual_time)
                time_div.append(div_time)
                time_diss.append(diss_time)
                time_cov.append(cov_time)
                time_score_exec.append(exec_time)

            else:
                score, info = compute_synthesis_score_faster(result,
                                                             self.code,
                                                             task,
                                                             self.ref_task,
                                                             ignore_diversity=ignore_diversity,
                                                             ignore_dissimilarity=ignore_dissimilarity,
                                                             compute_visitation_quality=visitation_quality)
            if TIME_DEBUG:
                end = time.time()
                time_score.append(end - start)

            if DEBUG:
                print(f"Hero row: {inp_world.heroRow}, hero col: {inp_world.heroCol}, "
                      f"hero dir: {inp_world.heroDir}, score: {score}")

            if score > max_score:
                if patience is not None and self.diff_threshold is not None:
                    if score - max_score < self.diff_threshold:
                        patience -= 1
                        if patience == 0:
                            print("Patience ran out. Stopping at iteration", i)
                            break
                    else:
                        patience = self.patience
                max_score = score
                best_inp_world = inp_world
                best_out_world = out_world
                best_info = info
                best_sym = result.outgrid
            elif patience is not None and self.diff_threshold is not None:
                patience -= 1
                if patience == 0:
                    print("Patience ran out. Stopping at iteration", i)
                    break

            if len(heap) < POOL_SIZE:
                heapq.heappush(heap, (score, i, inp_world,
                                      out_world, info))
            else:
                # Equivalent to a push, then a pop, but faster
                _ = heapq.heappushpop(heap, (score, i, inp_world,
                                             out_world, info))

            time_lst.append(time.time() - start_all)

            if TIME_DEBUG:
                start = time.time()
            if log_freq and i % log_freq == 0:
                logging.info(f"Current best score: {max_score}")
                logging.info(f"Current best info: {best_info}")

                if not best_info:
                    # if log_freq:
                        # wandb.log({f"{log_prefix}score": max_score})
                    continue

            if TIME_DEBUG:
                end = time.time()
                time_log.append(end - start)

        if TIME_DEBUG:
            print(f"Time log: {avg(time_log)}")
            print(f"Time from worlds: {avg(time_from_worlds)}")
            print(f"Time world conv: {avg(time_world_conv)}")
            print(f"Time score: {avg(time_score)}")
            print(f"    Time solv: {avg(time_solv)}")
            print(f"    Time qual: {avg(time_qual)}")
            print(f"    Time div: {avg(time_div)}")
            print(f"    Time diss: {avg(time_diss)}")
            print(f"    Time cov: {avg(time_cov)}")
            print(f"    Time exec: {avg(time_score_exec)}")
            print(f"    Time total score: "
                  f"{avg(time_solv) + avg(time_qual) + avg(time_div) + avg(time_diss) + avg(time_cov) + avg(time_score_exec)}")
            print(f"Time emulation: {avg(time_emulation)}")
            print(f"Time suspects: ",
                  avg(time_score) + avg(time_emulation) + avg(time_world_conv))
            print(f"Time lst: {avg(time_lst)}")
            print(f"Total time: {sum(time_lst)}")

        # if log_freq:
            # wandb.log({f"{log_prefix}total_time": sum(time_lst),
            #           f"{log_prefix}avg_time": avg(time_lst)})

        score, evaluation = \
            compute_evaluation_score(self.code,
                                     Task(
                                         self.init_task.pregrids + [best_inp_world],
                                         self.init_task.postgrids + [best_out_world],
                                         type_=self.code.type),
                                     self.ref_task,
                                     ignore_diversity=ignore_diversity,
                                     ignore_dissimilarity=ignore_dissimilarity,
                                     compute_visitation_quality=visitation_quality,
                                     compute_shortest_path_quality=compute_shortest_path_quality)

        if not any([evaluation['redundancy'] == 'NOT RESPECTED',
                    evaluation['solvability'] == 'NOT RESPECTED',
                    evaluation['shortest_path'] == 'NOT RESPECTED',
                    evaluation['coverage'] != 1.0]):
            if evaluation['vis_score'] == 1.0:
                return best_inp_world, best_out_world, score, heap

        for _, _, inp_world, out_world, info in heapq.nlargest(POOL_SIZE, heap):
            score, evaluation = \
                compute_evaluation_score(self.code,
                                         Task(
                                             self.init_task.pregrids + [inp_world],
                                             self.init_task.postgrids + [out_world],
                                             type_=self.code.type),
                                         self.ref_task,
                                         ignore_diversity=ignore_diversity,
                                         ignore_dissimilarity=ignore_dissimilarity,
                                         compute_visitation_quality=visitation_quality,
                                         compute_shortest_path_quality=compute_shortest_path_quality)

            if not any([evaluation['redundancy'] == 'NOT RESPECTED',
                        evaluation['solvability'] == 'NOT RESPECTED',
                        evaluation['shortest_path'] == 'NOT RESPECTED',
                        evaluation['coverage'] != 1.0]):
                if evaluation['vis_score'] == 1.0:
                    return inp_world, out_world, score, heap

        raise ValueError("No valid world found")


def obtain_karel_saturation_score_for_code(code,
                                           decision_maker=RandomDecisionMaker.auto_init(),
                                           max_iterations=100_000,
                                           patience=None, diff_threshold=None):
    assert code.type == 'karel'

    ref_task = Task([], [], 'karel')
    post_processor = BlockedPostProcessor()

    score = [0.0]
    task = None
    try:
        task_synthesizer = TaskSynthesizer(code, ref_task,
                                           decision_maker, post_processor,
                                           max_iterations=max_iterations,
                                           patience=patience,
                                           diff_threshold=diff_threshold)
        task, score, _ = task_synthesizer.synthesize(1, 50, ','.join(blocktypes),
                                                     log_freq=None,
                                                     type_='karel',
                                                     init_symworlds=None,
                                                     ignore_diversity=True,
                                                     ignore_dissimilarity=True,
                                                     compute_shortest_path_quality=True,
                                                     visitation_quality=True
                                                     )
    except Exception:
        pass

    return score, task


def obtain_hoc_saturation_score_for_code(code,
                                         decision_maker=RandomDecisionMaker.auto_init(),
                                         max_iterations=100_000,
                                         patience=None, diff_threshold=None):
    assert code.type == 'hoc'

    ref_task = Task([], [], 'hoc')
    post_processor = BlockedPostProcessor()

    score = [0.0]
    task = None
    try:
        task_synthesizer = TaskSynthesizer(code, ref_task,
                                           decision_maker, post_processor,
                                           max_iterations=max_iterations,
                                           patience=patience,
                                           diff_threshold=diff_threshold)
        task, score, heaps = task_synthesizer.synthesize(1, 50, ','.join(blocktypes),
                                                         log_freq=None,
                                                         type_='hoc',
                                                         init_symworlds=None,
                                                         ignore_diversity=True,
                                                         ignore_dissimilarity=True,
                                                         # compute_shortest_path_quality=True,)
                                                         visitation_quality=True)
    except Exception as e:
        # print(e)
        pass

    return score, task


def obtain_hoc_saturation_score_for_code_multiple_runs(code,
                                                       iteration_list,
                                                       decision_maker=RandomDecisionMaker.auto_init(),
                                                       ):
    assert code.type == 'hoc'

    ref_task = Task([], [], 'hoc')
    post_processor = BlockedPostProcessor()

    sorted_iteration_list = sorted(iteration_list)

    scores = []
    score = [0.0]
    heap = None
    # print('Starting with iteration', sorted_iteration_list[0])
    try:
        task_synthesizer = TaskSynthesizer(code, ref_task,
                                           decision_maker, post_processor,
                                           max_iterations=sorted_iteration_list[0])
        task, score, heaps = task_synthesizer.synthesize(1, 50, ','.join(blocktypes),
                                                         log_freq=None,
                                                         type_='hoc',
                                                         init_symworlds=None,
                                                         ignore_diversity=True,
                                                         ignore_dissimilarity=True,
                                                         # compute_shortest_path_quality=True,
                                                         visitation_quality=True)
    except Exception as e:
        # print(e)
        # print(traceback.format_exc())
        pass

    scores.append(score[0])
    prev_score = score

    for i in range(1, len(sorted_iteration_list)):
        try:
            task_synthesizer = TaskSynthesizer(code, ref_task,
                                               decision_maker, post_processor,
                                               max_iterations=sorted_iteration_list[i] - sorted_iteration_list[i - 1]
                                               )
            task, score, heap = task_synthesizer.synthesize(1, 50, ','.join(blocktypes),
                                                            log_freq=None,
                                                            type_='hoc',
                                                            init_symworlds=None,
                                                            ignore_diversity=True,
                                                            ignore_dissimilarity=True,
                                                            # compute_shortest_path_quality=True,)
                                                            visitation_quality=True)
            if score[0] > prev_score[0]:
                prev_score = score
        except Exception:
            pass
        scores.append(prev_score[0])

    final_scores = {}

    for idx, x in enumerate(sorted_iteration_list):
        final_scores[x] = scores[idx]

    return final_scores


def obtain_karel_saturation_score_for_code_multiple_runs(code,
                                                         iteration_list,
                                                         decision_maker=RandomDecisionMaker.auto_init(),
                                                         ):
    assert code.type == 'karel'

    ref_task = Task([], [], 'karel')
    post_processor = BlockedPostProcessor()

    sorted_iteration_list = sorted(iteration_list)

    scores = []
    score = [0.0]
    heap = None
    try:
        task_synthesizer = TaskSynthesizer(code, ref_task,
                                           decision_maker, post_processor,
                                           max_iterations=sorted_iteration_list[0])
        task, score, heaps = task_synthesizer.synthesize(1, 50, ','.join(blocktypes),
                                                         log_freq=None,
                                                         type_='karel',
                                                         init_symworlds=None,
                                                         ignore_diversity=True,
                                                         ignore_dissimilarity=True,
                                                         compute_shortest_path_quality=True,
                                                         visitation_quality=True)
    except Exception:
        pass

    scores.append(score[0])
    prev_score = score

    for i in range(1, len(sorted_iteration_list)):
        try:
            task_synthesizer = TaskSynthesizer(code, ref_task,
                                               decision_maker, post_processor,
                                               max_iterations=sorted_iteration_list[i] - sorted_iteration_list[i - 1]
                                               )
            task, score, heap = task_synthesizer.synthesize(1, 50, ','.join(blocktypes),
                                                            log_freq=None,
                                                            type_='karel',
                                                            init_symworlds=None,
                                                            ignore_diversity=True,
                                                            ignore_dissimilarity=True,
                                                            compute_shortest_path_quality=True,
                                                            visitation_quality=True)
            if score[0] > prev_score[0]:
                prev_score = score
        except Exception:
            pass
        scores.append(prev_score[0])

    final_scores = {}

    for idx, x in enumerate(sorted_iteration_list):
        final_scores[x] = scores[idx]

    return final_scores
