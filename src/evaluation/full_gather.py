import argparse
import json
import os

import numpy as np


def full_succ_ratios(type_, perc, agg, seeds):
    for difficulty in ['all', 'easy', 'hard']:
        print("DIFFICULTY: ", difficulty)
        for model in ['base', 'neural']:
            print("MODEL: ", model)
            succ_ratios = []
            for seed in seeds:
                results = []
                for run in range(agg):
                    succs = []
                    if model == 'neural':
                        path = f'results/{type_}/seed_{seed}/Full_LSTMDecisionMaker_GridActorCriticLookaheadDecisionMaker.json'
                    elif model == 'base':
                        path = f'results/{type_}/seed_{seed}/Full_RandomCodeDecisionMaker_RandomDecisionMaker.json'
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

                            ok = False
                            for index, res in enumerate(line['codegen_results']):
                                for sc in res['taskgen_scores_noncumulative']:
                                    if sc > perc*res['oracle_score'] and res['oracle_score'] > 0:
                                        ok = True
                            if ok:
                                succs.append(1)
                            else:
                                succs.append(0)
                    results.append(np.mean(succs))
                    # print(f"Size for seed {seed}: ", len(succs))
                succ_ratios.append(np.mean(results))

            # multiply by 100
            succ_ratios = [x*100 for x in succ_ratios]
            # print with two decimal places
            print(f"mean: {np.mean(succ_ratios):.2f}")
            if len(seeds) > 1:
                print(f"std err: {np.std(succ_ratios, ddof=1) / np.sqrt(len(succ_ratios)):.2f}")
            else:
                print("std err: n/a (only one seed)")

        print()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--domain', type=str, default='hoc')
    # seeds
    arg_parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    args = arg_parser.parse_args()

    # check if results file exist for each seed
    for seed in args.seeds:
        if not os.path.exists(f'results/{args.domain}/seed_{seed}/Full_LSTMDecisionMaker_GridActorCriticLookaheadDecisionMaker.json'):
            raise ValueError(f"Results file for neural for seed {seed} does not exist")
        if not os.path.exists(f'results/{args.domain}/seed_{seed}/Full_RandomCodeDecisionMaker_RandomDecisionMaker.json'):
            raise ValueError(f"Results file for base for seed {seed} does not exist")

    type_ = args.domain

    perc = 0.9
    agg = 1

    full_succ_ratios(type_, perc, agg, args.seeds)