import argparse
import random
from typing import List

import numpy as np
import torch
# import wandb
# from decouple import config
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.codetask_scoring.finalscore import compute_synthesis_score_faster, \
    compute_evaluation_score
from src.emulator.code import Code
from src.emulator.fast_emulator import FastEmulator
from src.emulator.task import Task
from src.symexecution.post_processor import BlockedPostProcessor, \
    EmptySpacePostProcessor
from src.symexecution.symworld import SymWorld
from src.taskgen.callbacks import Callback, \
    SavingCallback, TaskSynthSuccRatioCallback
from src.taskgen.data import CodeDataset, DifficultCodeDataset
from src.taskgen.decision_makers import GridActorCriticLookaheadDecisionMaker
from src.taskgen.feature_processors import get_features_size
from src.taskgen.networks import GridActionValueHandmadeFeaturesNetwork

MAX_WORLD_SIZE = (16, 16)


# API_KEY = config("API_KEY")


def train(agent: GridActorCriticLookaheadDecisionMaker,
          emulator: FastEmulator,
          dataloader_default: DataLoader,
          callbacks: List[Callback],
          optimizers: List[optim.Optimizer],
          n_epochs: int,
          scoring_function='evaluation',
          schedulers=None,
          segments_weight=None,
          dataloader_difficult=None,
          difficult_dataset_prob=None,
          curriculum=False
          ):
    agent.train()

    callback_args = {'epoch': -1}
    # # wandb.log({'epoch': -1})
    for callback in callbacks:
        result = callback.execute(**callback_args)
        if result is not None:
            for val, name in result:
                print(f"{name}: {val}")
                # wandb.log({name: val})
                callback_args[name] = val

    print("Training...")
    for epoch in tqdm(range(n_epochs)):
        epoch_rewards = []
        if difficult_dataset_prob is None:
            dataloader = dataloader_default
        else:
            # sample randomly one dataloader based on difficult_dataset_prob
            dataloader = dataloader_default if np.random.rand() > difficult_dataset_prob * (epoch / n_epochs) \
                else dataloader_difficult

        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            agent.set_emulator(emulator)

            agent.reset_buffer()

            for ex_idx, (example, code_type, sat_score) in enumerate(batch):
                code = Code.parse_json(example)
                code = agent.process_code(code)

                if "ref_task" in example:
                    ref_task = Task.parse_json(example["ref_task"])
                else:
                    ref_task = Task([], [], code.type)

                if 'buffer' in example:
                    agent.set_current_example(example['buffer'])

                if 'rows' in example and 'cols' in example:
                    rows = example['rows']
                    cols = example['cols']
                elif 'examples' in example:
                    rows = example['examples'][0]['inpgrid_json']['rows']
                    cols = example['examples'][0]['inpgrid_json']['cols']
                else:
                    rows, cols = MAX_WORLD_SIZE

                symworld = SymWorld.empty_init(rows, cols, agent)
                if isinstance(agent, GridActorCriticLookaheadDecisionMaker):
                    agent.set_code_type(code_type)
                res = emulator.emulate(code, symworld)

                if code.type == "hoc":
                    post_processor = BlockedPostProcessor()
                else:
                    post_processor = EmptySpacePostProcessor()

                inp_world, out_world = post_processor.symworld_to_world(
                    res.outgrid)

                task = Task([inp_world],
                            [out_world],
                            type_=code.type)

                if scoring_function == 'synthesis':
                    score, info = compute_synthesis_score_faster(res, code, task,
                                                                 ref_task)
                elif scoring_function == 'evaluation':
                    score, info = compute_evaluation_score(code,
                                                           task,
                                                           ref_task,
                                                           compute_visitation_quality=True,
                                                           compute_shortest_path_quality=True,
                                                           ignore_diversity=True,
                                                           ignore_dissimilarity=True,
                                                           segments_weight=segments_weight
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

                    if curriculum:
                        if epoch < 2 * n_epochs / 10:
                            perc = 0
                        # increase gradually from 0.8 to 0.9 in the next 2 * n_epochs / 10 epochs
                        elif epoch < 6 * n_epochs / 10:
                            perc = 0.8 + (epoch - 2 * n_epochs / 10) / (4 * n_epochs)
                            if score < perc * sat_score[0]:
                                score = 0.00
                        else:
                            perc = 0.9
                            if score < perc * sat_score[0]:
                                score = 0.00

                        # wandb.log({'curriculum': perc})

                agent.populate_rewards(score)

            rets = agent.compute_loss()

            ret = rets[-1]
            losses = rets[:-1]

            for optimizer in optimizers:
                optimizer.zero_grad()
            for loss in losses:
                loss.backward()

            for optimizer in optimizers:
                optimizer.step()

            epoch_rewards.append(np.mean(ret))
            # for i, loss in enumerate(losses):
            # wandb.log({f"loss_{i}": loss.item()})

        epoch_reward = sum(epoch_rewards) / len(epoch_rewards)
        # wandb.log({'epoch_reward': epoch_reward})
        print(f"Epoch {epoch} reward: {epoch_reward}")

        callback_args = {'epoch': epoch}
        # wandb.log({'epoch': epoch})
        for callback in callbacks:
            result = callback.execute(**callback_args)
            if result is not None:
                for val, name in result:
                    print(f"{name}: {val}")
                    # wandb.log({name: val})
                    callback_args[name] = val

        if schedulers is not None:
            for scheduler in schedulers:
                scheduler.step()

        # wandb.log({'lr': optimizers[0].param_groups[0]['lr']})

    return agent


if __name__ == '__main__':
    print("WARNING: experiment with large waiting time")

    # make arguments for the script
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--domain', type=str, default='hoc')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--grid_stack', type=int, nargs='+', default=[1024, 512, 256, 128, 32])
    parser.add_argument('--decision_stack', type=int, nargs='+', default=[8])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--cnn_layer_sizes', type=int, nargs='+', default=[64, 64, 64])
    parser.add_argument('--pooling_layer_sizes', type=int, nargs='+', default=[2, 2, 2])
    parser.add_argument('--temperature', type=float, default=1)

    args = parser.parse_args()
    seed = args.seed

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    type_ = args.domain

    learning_rate = args.learning_rate  # 1e-4
    actor_fc_stack = args.grid_stack  # [1024, 512, 256, 128, 32]
    decision_stack = args.decision_stack  # [8]
    batch_size = args.batch_size  # 32
    n_epochs = args.epochs  # 500
    cnn_layer_sizes = [(x, 3, 1, 1, 1) for x in
                       args.cnn_layer_sizes]  # ((64, 3, 1, 1, 1), (64, 3, 1, 1, 1), (64, 3, 1, 1, 1))
    pooling_layer_sizes = [(x, 2, 1, 0) for x in
                           args.pooling_layer_sizes]  # ((2, 2, 1, 0), (2, 2, 1, 0), (2, 2, 1, 0))
    temperature = args.temperature  # 1

    eval_rollouts = 10
    eval_runs = 10
    eval_sample_size = 25

    add_new = True

    diff_sk_start = 5 if type_ == 'hoc' else 7
    curriculum = True

    dir_path = f'data/synthetic/{type_}/seed_{seed}'

    if type_ == 'hoc':
        preprocess_type = "grid_and_coverage_action_count_pre_process_input_hoc"
    elif type_ == 'karel':
        preprocess_type = "grid_and_coverage_action_count_pre_process_input_compact_karel_markerbitmap"
    else:
        raise Exception("Unknown type")

    evaluation_every_n_epochs = 5

    emulator = FastEmulator(1000, 1000)

    network = GridActionValueHandmadeFeaturesNetwork(
        input_size=get_features_size(preprocess_type),
        cnn_layer_sizes=cnn_layer_sizes,
        pooling_layers_sizes=pooling_layer_sizes,
        latent_layer_sizes=actor_fc_stack,
        output_size=1,
        append_features=True,
        batch_norm=False,
        decision_layers=decision_stack,
    )

    training_dataset = CodeDataset(f'{dir_path}/train.json')
    if add_new:
        new_training_dataset = CodeDataset(f'{dir_path}/codes_100.json')
        training_dataset = training_dataset + new_training_dataset

    training_dataset_difficult = DifficultCodeDataset(f'{dir_path}/train.json', sk_nb=diff_sk_start)
    if add_new:
        new_training_dataset_difficult = DifficultCodeDataset(f'{dir_path}/codes_100.json', sk_nb=diff_sk_start)
        training_dataset_difficult = training_dataset_difficult + new_training_dataset_difficult

    validation_dataset = CodeDataset(f'{dir_path}/val.json')
    difficult_val_dataset = DifficultCodeDataset(f'{dir_path}/val.json', sk_nb=diff_sk_start)

    decision_maker = GridActorCriticLookaheadDecisionMaker(
        network=network,
        preprocess_function=preprocess_type,
        emulator=emulator,
        has_buffer=True,
        temperature=temperature,
        entropy=None)

    save_path = f'models/{type_}/taskgen/seed_{seed}/new'

    # config = {
    #     "learning_rate": learning_rate,
    #     "actor_fc_stack": actor_fc_stack,
    #     "batch_size": batch_size,
    #     "n_epochs": n_epochs,
    #     "preprocess_type": preprocess_type,
    #     "evaluation_every_n_epochs": evaluation_every_n_epochs,
    #     "cnn_layer_sizes": cnn_layer_sizes,
    #     "pooling_layer_sizes": pooling_layer_sizes,
    #     "alpha": alpha,
    #     "save_path": save_path,
    #     "eval_rollouts": eval_rollouts,
    #     "eval_runs": eval_runs,
    #     "eval_sample_size": eval_sample_size,
    #     "difficult_sk_start": diff_sk_start,
    #     "curriculum": curriculum,
    #     "decision_stack": decision_stack,
    #     "temperature": temperature,
    # }
    #
    # # wandb.login(key=API_KEY)
    # wandb.init(project="symworld-decision-makers",
    #            name=f"Augmented_Seed_{seed}_{nb}",
    #            group=f"{type_}-Seeds",
    #            config=config,
    #            mode="online")

    optimizer = optim.Adam(decision_maker.get_parameters(),
                           lr=learning_rate)

    dataloader = DataLoader(training_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=lambda x: x)
    dataloader_difficult = DataLoader(training_dataset_difficult,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      collate_fn=lambda x: x)
    eval_callback = TaskSynthSuccRatioCallback(evaluation_every_n_epochs,
                                               decision_maker,
                                               validation_dataset,
                                               no_rollouts=eval_rollouts,
                                               no_runs=eval_runs,
                                               sample_size=eval_sample_size,
                                               type_=type_
                                               )

    difficult_eval_callback = TaskSynthSuccRatioCallback(evaluation_every_n_epochs,
                                                         decision_maker,
                                                         difficult_val_dataset,
                                                         no_rollouts=eval_rollouts,
                                                         no_runs=eval_runs,
                                                         sample_size=eval_sample_size,
                                                         type_=type_,
                                                         prefix='difficult'
                                                         )

    saving_callback = SavingCallback(decision_maker,
                                     save_path)

    train(agent=decision_maker,
          dataloader_default=dataloader,
          callbacks=[eval_callback, difficult_eval_callback, saving_callback],
          emulator=emulator,
          optimizers=[optimizer],
          n_epochs=n_epochs,
          schedulers=None,
          dataloader_difficult=dataloader_difficult,
          curriculum=curriculum)
