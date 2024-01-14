import json
import time
from typing import Tuple

from src.codetask_scoring.coverage import compute_coverage, \
    compute_coverage_from_executor_result, compute_coverage_from_emulator_result
from src.codetask_scoring.deltadebugging import \
    check_codetask_redundancy_and_delta
from src.codetask_scoring.execquality import compute_codetask_quality, \
    compute_codetask_quality_from_executor_result, \
    compute_codetask_quality_from_emulator_result, \
    compute_visitation_quality_from_executor_result, \
    compute_visitation_quality_from_emulator_result
from src.codetask_scoring.shortestpath import check_shortest_path
from src.codetask_scoring.solvability import check_solvability, \
    check_solvability_from_executor_result, check_solvability_from_emulator_result, \
    check_code_sanity
from src.codetask_scoring.task_dissimilarity import compute_task_dissimilarity, \
    compute_task_diversity
from src.emulator.code import Code
from src.emulator.executor import Executor
from src.emulator.fast_emulator import EmuResult
from src.emulator.task import Task
from src.emulator.tokens import actions

DELTA_QUALITY = 0.2

TIME_DEBUG = False


def compute_final_score(code: Code, task: Task, reference_task: Task,
                        full_task_dissimilarity: bool = False) -> Tuple[float,
                                                                        dict]:
    """Compute the final score for a code snippet for a given task."""
    if not check_solvability(code, task):
        return 0.0, {}

    if not check_shortest_path(code, task):
        return 0.0, {}

    score = 0.0
    norm = 0

    info = {}

    # 0.003 seconds
    ########################################
    aux_score, aux_info = compute_codetask_quality(code, task)
    score += aux_score
    norm += 1
    info['quality'] = aux_info

    if aux_score < DELTA_QUALITY:
        return 0.0, {}
    ########################################

    # 0.001 seconds
    ########################################
    if reference_task is not None and reference_task.num_examples > 0:
        aux_score, diss_info = compute_task_dissimilarity(task, reference_task,
                                                          full_task_dissimilarity)
        score += aux_score
        norm += 1
    else:
        diss_info = [{'loc_diss': 1,
                      'dir_diss': 1,
                      'grid_diss': 1}]
    info['dissimilarity'] = diss_info
    ########################################

    # 0.004 seconds
    ########################################
    aux_score = compute_coverage(code, task)
    score += aux_score
    norm += 1
    info['coverage'] = aux_score
    ########################################

    return score / norm, info

def compute_synthesis_score_fast(code: Code, task: Task, reference_task: Task,
                                 full_task_dissimilarity: bool = False):
    if TIME_DEBUG:
        solv_time = []
        qual_time = []
        cov_time = []
        diss_time = []
        div_time = []
        exec_time = []

    if TIME_DEBUG:
        start = time.time()
    executor = Executor()
    result = executor.execute(task, code)
    if TIME_DEBUG:
        exec_time.append(time.time() - start)

    if TIME_DEBUG:
        start = time.time()
    if not check_solvability_from_executor_result(result):
        if TIME_DEBUG:
            solv_time.append(time.time() - start)
            return 0.0, {}, solv_time[0], 0.0, 0.0, 0.0, 0.0, exec_time[0]

        return 0.0, {}
    if TIME_DEBUG:
        solv_time.append(time.time() - start)

    # if not check_shortest_path(code, task):
    #     return 0.0, {}

    score = 0.0
    norm = 0

    info = {}

    if TIME_DEBUG:
        start = time.time()
    quality_score, aux_info = \
        compute_codetask_quality_from_executor_result(result,
                                                      [max(world.rows, world.cols)
                                                       for world in task.pregrids],
                                                      task.type)
    if TIME_DEBUG:
        qual_time.append(time.time() - start)

    score += quality_score
    norm += 1
    info['quality'] = aux_info

    # if aux_score < DELTA_QUALITY:
    #     return 0.0, {}

    if reference_task is not None and reference_task.num_examples > 0:
        if TIME_DEBUG:
            start = time.time()
        dissimilarity_score, diss_info = \
            compute_task_dissimilarity(task,
                                       reference_task,
                                       full_task_dissimilarity)
        if TIME_DEBUG:
            diss_time.append(time.time() - start)
        score += dissimilarity_score
        norm += 1
    else:
        diss_info = [{'loc_diss': 1,
                      'dir_diss': 1,
                      'grid_diss': 1}]
    info['dissimilarity'] = diss_info

    if TIME_DEBUG:
        start = time.time()
    diversity_score, info_ = compute_task_diversity(task)
    if TIME_DEBUG:
        div_time.append(time.time() - start)
    score += diversity_score
    norm += 1
    info['diversity'] = info_

    if TIME_DEBUG:
        start = time.time()
    coverage_score = compute_coverage_from_executor_result(result, code)
    if TIME_DEBUG:
        cov_time.append(time.time() - start)
    score += coverage_score
    norm += 1
    info['coverage'] = coverage_score

    if TIME_DEBUG:
        return score / norm, info, solv_time[0], qual_time[0], diss_time[0], \
               div_time[0], \
               cov_time[0], exec_time[0]

    return score / norm, info


def compute_synthesis_score_faster(result: EmuResult, code: Code, task: Task,
                                   reference_task: Task,
                                   full_task_dissimilarity: bool = False,
                                   ignore_diversity: bool = False,
                                   ignore_dissimilarity: bool = False,
                                   compute_visitation_quality: bool = False
                                   ):
    if TIME_DEBUG:
        solv_time = []
        qual_time = []
        cov_time = []
        diss_time = []
        div_time = []
        exec_time = []

    if TIME_DEBUG:
        start = time.time()
    # executor = Executor()
    # result = executor.execute(task, code)
    if TIME_DEBUG:
        exec_time.append(time.time() - start)

    if TIME_DEBUG:
        start = time.time()
    if not check_solvability_from_emulator_result(result):
        if TIME_DEBUG:
            solv_time.append(time.time() - start)
            return 0.0, {}, solv_time[0], 0.0, 0.0, 0.0, 0.0, exec_time[0]

        return 0.0, {}
    if TIME_DEBUG:
        solv_time.append(time.time() - start)

    # if not check_shortest_path(code, task):
    #     return 0.0, {}

    score = 0.0
    norm = 0

    info = {}

    if TIME_DEBUG:
        start = time.time()
    quality_score, aux_info = \
        compute_codetask_quality_from_emulator_result(result,
                                                      max(task.pregrids[-1].rows,
                                                          task.pregrids[-1].cols),
                                                      task.type)
    if TIME_DEBUG:
        qual_time.append(time.time() - start)

    score += quality_score
    norm += 1
    info['quality'] = aux_info

    # if aux_score < DELTA_QUALITY:
    #     return 0.0, {}

    if reference_task is not None and\
            not ignore_dissimilarity and \
            reference_task.num_examples > 0:
        if TIME_DEBUG:
            start = time.time()
        dissimilarity_score, diss_info = \
            compute_task_dissimilarity(task,
                                       reference_task,
                                       full_task_dissimilarity)
        if TIME_DEBUG:
            diss_time.append(time.time() - start)
        score += dissimilarity_score
        norm += 1
    else:
        diss_info = [{'loc_diss': -1,
                      'dir_diss': -1,
                      'grid_diss': -1}]
    info['dissimilarity'] = diss_info

    if not ignore_diversity:
        if TIME_DEBUG:
            start = time.time()
        diversity_score, info_ = compute_task_diversity(task)
        if TIME_DEBUG:
            div_time.append(time.time() - start)

        score += diversity_score
        norm += 1
        info['diversity'] = info_

    if TIME_DEBUG:
        start = time.time()
    coverage_score = compute_coverage_from_emulator_result(result, code)
    if TIME_DEBUG:
        cov_time.append(time.time() - start)
    score += coverage_score
    norm += 1
    info['coverage'] = coverage_score

    if not ignore_diversity and diversity_score == 0:
        score = 0

    if TIME_DEBUG:
        return score / norm, info, solv_time[0], qual_time[0], diss_time[0], \
               div_time[0], \
               cov_time[0], exec_time[0]

    if compute_visitation_quality:
        vis_score = compute_visitation_quality_from_emulator_result(result)
        if vis_score < 0.9:
            score /= 2
        info['visitation_quality'] = vis_score

    return score / norm, info


def compute_evaluation_score(code: Code, task: Task, reference_task: Task,
                             full_task_dissimilarity: bool = False,
                             for_entry: int = None,
                             compute_shortest_path_quality: bool = False,
                             compute_visitation_quality: bool = False,
                             ignore_diversity: bool = False,
                             ignore_dissimilarity: bool = False,
                             segments_weight=None) \
        -> Tuple[float, dict]:
    executor = Executor()
    result = executor.execute(task, code)

    info = {}

    if not check_solvability_from_executor_result(result):
        info['solvability'] = 'NOT RESPECTED'
    else:
        info['solvability'] = 'RESPECTED'

    if not check_code_sanity(code):
        info['sanity'] = 'NOT RESPECTED'
    else:
        info['sanity'] = 'RESPECTED'

    # 0.086 seconds
    ########################################
    if check_codetask_redundancy_and_delta(code, task,
                                           keep_empty_body=False,
                                           unwrap=True):
        info['redundancy'] = 'NOT RESPECTED'
    else:
        info['redundancy'] = 'RESPECTED'
    ########################################

    simple_code = len(set([x for x in code.block_count
                           if code.block_count[x] != 0]) -
                      set(actions)) == 0

    score = 0.0
    norm = 0

    shortest_path, shortest_path_quality = check_shortest_path(code, task,
                                                               check_path_quality=True)
    if not shortest_path and not simple_code:
        info['shortest_path'] = 'NOT RESPECTED'
    else:
        info['shortest_path'] = 'RESPECTED'

    # if compute_shortest_path_quality and code.type == 'hoc':
    if compute_shortest_path_quality and shortest_path_quality is not None:
        score += sum(shortest_path_quality) / len(shortest_path_quality)
        norm += 1
        info['shortest_path_quality'] = shortest_path_quality

    # 0.003 seconds
    ########################################
    aux_score, aux_info = \
        compute_codetask_quality_from_executor_result(result,
                                                      [max(world.rows, world.cols)
                                                       for world in task.pregrids],
                                                      task.type,
                                                      segments_weight=segments_weight)
    score += aux_score
    norm += 1
    info['quality'] = aux_info

    if aux_score < DELTA_QUALITY:
        info['quality_delta'] = 'BELOW DELTA'
    else:
        info['quality_delta'] = 'ABOVE DELTA'
    ########################################

    # 0.001 seconds
    ########################################
    if not ignore_dissimilarity:
        if reference_task is not None and reference_task.num_examples > 0:
            aux_score, diss_info = compute_task_dissimilarity(task, reference_task,
                                                              full_task_dissimilarity,
                                                              for_entry)
            score += aux_score
            norm += 1
        else:
            diss_info = [{'loc_diss': 1,
                          'dir_diss': 1,
                          'grid_diss': 1}]
        info['dissimilarity'] = diss_info
    ########################################

    if not ignore_diversity:
        aux_score, info_ = compute_task_diversity(task,
                                                  full_task_dissimilarity,
                                                  for_entry)
        score += aux_score
        norm += 1
        info['diversity'] = info_

    # 0.004 seconds
    ########################################
    aux_score = compute_coverage_from_executor_result(result, code)
    score += aux_score
    norm += 1
    info['coverage'] = aux_score
    ########################################

    # if compute_visitation_quality and code.type == 'hoc':
    if compute_visitation_quality:
        visitation_quality = compute_visitation_quality_from_executor_result(result)
        vis_score = sum(visitation_quality) / len(visitation_quality)
        # if vis_score < 0.9:
        #     score /= 2
        info['visitation_quality'] = visitation_quality
        info['vis_score'] = vis_score

    return score / norm, info