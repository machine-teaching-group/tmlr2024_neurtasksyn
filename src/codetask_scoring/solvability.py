from src.emulator.code import Code
from src.emulator.executor import Executor
from src.emulator.fast_emulator import EmuResult
from src.emulator.task import Task


def check_solvability(code: Code, task: Task) -> bool:
    """
    Given a single code-task pair returns solvability(i.e., grids are solved for the given code) True/False
    """

    executor = Executor()

    result = executor.execute(task, code)
    return result['task_success']


def check_solvability_from_executor_result(result: dict) -> bool:
    return result['task_success']


def check_solvability_from_emulator_result(result: EmuResult) -> bool:
    return not result.crashed and not result.timeout


def check_code_sanity(code: Code) -> bool:
    """
    Checks if the code is valid
    """
    if code.block_count['repeatUntil'] > 0:
        children = code.astJson['body']
        children_types = [child['type'] for child in children]
        if sum(['repeatUntil' == child_type for child_type in children_types]) > 1:
            return False
    return True
