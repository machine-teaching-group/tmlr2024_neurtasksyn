import argparse
import json
import os

import numpy as np

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--algo_type', type=str, default='neural', choices=['base', 'neural'])
    arg_parser.add_argument('--percentage', type=int, default=99)

    args = arg_parser.parse_args()
    algo = args.algo_type
    percentage = args.percentage

    path = f'results/time_measurements_{algo}_quality_{percentage}.json'

    if not os.path.exists(path):
        raise ValueError(f"Results file for {algo} does not exist")

    with open(path, 'r') as f:
        data = json.load(f)

    easy_rolls = []
    hard_rolls = []
    all_rolls = []
    easy_time = []
    hard_time = []
    all_time = []
    for i, d in enumerate(data):
        if d['dff'] == 'easy':
            easy_rolls.append([])
            easy_time.append([])
        else:
            hard_rolls.append([])
            hard_time.append([])

        all_rolls.append([])
        all_time.append([])

        for j, trial in enumerate(d['trials']):
            if d['dff'] == 'easy':
                easy_time[-1].append(trial['time'])
                easy_rolls[-1].append(trial['total_rollouts'])
            else:
                hard_time[-1].append(trial['time'])
                hard_rolls[-1].append(trial['total_rollouts'])

            all_time[-1].append(trial['time'])
            all_rolls[-1].append(trial['total_rollouts'])

    easy_rolls = np.array(easy_rolls, dtype=object)
    hard_rolls = np.array(hard_rolls, dtype=object)
    all_rolls = np.array(all_rolls, dtype=object)
    easy_time = np.array(easy_time, dtype=object)
    hard_time = np.array(hard_time, dtype=object)
    all_time = np.array(all_time, dtype=object)

    # compute average on outside list
    easy_rolls_avg = np.mean(easy_rolls, axis=0)
    hard_rolls_avg = np.mean(hard_rolls, axis=0)
    all_rolls_avg = np.mean(all_rolls, axis=0)
    easy_time_avg = np.mean(easy_time, axis=0)
    hard_time_avg = np.mean(hard_time, axis=0)
    all_time_avg = np.mean(all_time, axis=0)

    # compute average and std err on inside list
    easy_rolls_avg_avg = np.mean(easy_rolls_avg)
    hard_rolls_avg_avg = np.mean(hard_rolls_avg)
    all_rolls_avg_avg = np.mean(all_rolls_avg)

    easy_rolls_avg_std = np.std(easy_rolls_avg, ddof=1) / np.sqrt(len(easy_rolls_avg))
    hard_rolls_avg_std = np.std(hard_rolls_avg, ddof=1) / np.sqrt(len(hard_rolls_avg))
    all_rolls_avg_std = np.std(all_rolls_avg, ddof=1) / np.sqrt(len(all_rolls_avg))

    easy_time_avg_avg = np.mean(easy_time_avg)
    hard_time_avg_avg = np.mean(hard_time_avg)
    all_time_avg_avg = np.mean(all_time_avg)

    easy_time_avg_std = np.std(easy_time_avg, ddof=1) / np.sqrt(len(easy_time_avg))
    hard_time_avg_std = np.std(hard_time_avg, ddof=1) / np.sqrt(len(hard_time_avg))
    all_time_avg_std = np.std(all_time_avg, ddof=1) / np.sqrt(len(all_time_avg))

    print(f"Quality {percentage}% - easy - rolls average: {easy_rolls_avg_avg:.2f}, std err: {easy_rolls_avg_std:.2f}")
    print(f"Quality {percentage}% - hard - rolls average: {hard_rolls_avg_avg:.2f}, std err: {hard_rolls_avg_std:.2f}")
    print(f"Quality {percentage}% - all - rolls average: {all_rolls_avg_avg:.2f}, std err: {all_rolls_avg_std:.2f}")
    print()
    print(f"Quality {percentage}% - easy - time average: {easy_time_avg_avg:.2f}, std err: {easy_time_avg_std:.2f}")
    print(f"Quality {percentage}% - hard - time average: {hard_time_avg_avg:.2f}, std err: {hard_time_avg_std:.2f}")
    print(f"Quality {percentage}% - all - time average: {all_time_avg_avg:.2f}, std err: {all_time_avg_std:.2f}")