import argparse
import json
import os

import numpy as np

if __name__ == '__main__':
    # get type_ as argument
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--domain', type=str, default='hoc')
    # seeds
    arg_parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    args = arg_parser.parse_args()

    # check if results file exist for each seed
    for seed in args.seeds:
        if not os.path.exists(f'results/{args.domain}/seed_{seed}/Task_Abl_GridActorCriticLookaheadDecisionMaker.json'):
            raise ValueError(f"Results file for neural for seed {seed} does not exist")
        if not os.path.exists(f'results/{args.domain}/seed_{seed}/Task_Abl_RandomDecisionMaker.json'):
            raise ValueError(f"Results file for base for seed {seed} does not exist")

    type_ = args.domain
    perc = 0.9

    rollouts = 10
    agg = 1

    for difficulty in ['all', 'easy', 'hard']:
        print("DIFFICULTY: ", difficulty)
        for model in ['base', 'neural']:
            print("MODEL: ", model)
            succ_ratios = []
            for seed in args.seeds:
                results = []
                for run in range(agg):
                    succs = []
                    if model == 'neural':
                        path = f'results/{type_}/seed_{seed}/Task_Abl_GridActorCriticLookaheadDecisionMaker.json'
                    elif model == 'base':
                        path = f'results/{type_}/seed_{seed}/Task_Abl_RandomDecisionMaker.json'
                    check_data_points = set()
                    alternate_oracle = {}
                    with open(path, 'r') as f:
                        for line in f:
                            try:
                                line = json.loads(line)
                            except json.decoder.JSONDecodeError:
                                continue
                            if line['run_nb'] != run:
                                continue
                            if difficulty == 'easy':
                                if type_ == 'hoc':
                                    if line['sketch_nb'] > 4:
                                        continue
                                elif type_ == 'karel':
                                    if line['sketch_nb'] > 6:
                                        continue
                            elif difficulty == 'hard':
                                if type_ == 'hoc':
                                    if line['sketch_nb'] < 5:
                                        continue
                                elif type_ == 'karel':
                                    if line['sketch_nb'] < 7:
                                        continue
                            elif difficulty == 'all':
                                pass
                            else:
                                raise ValueError(f"Unknown difficulty {difficulty}")
                            if line['oracle_score'][0] == 0:
                                succs.append(0)
                                continue
                            for i, score in enumerate(line['taskgen_scores_noncumulative']):
                                if i >= rollouts:
                                    succs.append(0)
                                    break
                                if score > perc*line['oracle_score'][0]:
                                    succs.append(1)
                                    break
                            else:
                                succs.append(0)

                    results.append(sum(succs) / len(succs))

                succ_ratios.append(sum(results) / len(results))

            # print(succ_ratios)
            # make percentages
            succ_ratios = [x * 100 for x in succ_ratios]
            # print with two decimal places
            print(f"mean: {np.mean(succ_ratios):.2f}")
            # std err
            if len(succ_ratios) > 1:
                sigma = np.std(succ_ratios, ddof=1)
                print(f"std err: {sigma / np.sqrt(len(succ_ratios)):.2f}")
            else:
                print("std err: n/a (only one seed)")
        print()
