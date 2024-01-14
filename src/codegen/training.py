import argparse
import copy
import json
import random

import Levenshtein
import numpy as np
import torch
# import wandb

# from decouple import config
from tqdm import tqdm

from src.emulator.code import Code
from src.codegen.callbacks import SavingCallback
from src.codegen.callbacks import CodeSuccessValCallback
from src.codegen.codegen import sketch2code
from src.codegen.decision_makers import LSTMDecisionMaker
from src.codegen.networks import LSTMNetwork
from src.codegen.utils import tok2idx
from src.codegen.data import CodeSpecAndSketchNbDataset


# API_KEY = config("API_KEY")


def train(
        epochs,
        decision_maker,
        train_dataloader,
        optimizer,
        callbacks,
        training_type='sup',
        add_latent=False,
        type_='hoc'
):
    assert training_type in ['sup', 'rl']

    callback_args = {'epoch': 0}
    # wandb.log({"epoch": 0})
    for callback in callbacks:
        result = callback.execute(**callback_args)
        if result is not None:
            for val, name in result:
                print(f"{name}: {val}")
                # wandb.log({name: val})
                callback_args[name] = val

    print("Training...")
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        losses = []
        for batch in tqdm(train_dataloader):
            for partial_sk, _, tgt, line, latent in batch:
                code = json.loads(line)
                code = Code(type_, code['program_json'])
                blocks = code.total_count
                # get random number between blocks and 16
                blocks = random.randint(blocks, 16)
                if training_type == 'sup':
                    sketch2code(copy.deepcopy(partial_sk), decision_maker, blocks, tgt=tgt,
                                latent=latent if add_latent else None,
                                type_=type_)
                elif training_type == 'rl':
                    _, lin_code = sketch2code(copy.deepcopy(partial_sk), decision_maker, blocks, type_=type_)
                    lev_loss = Levenshtein.distance(lin_code, tgt)
                    decision_maker.do_end_code(lev_loss)
                else:
                    raise ValueError('Invalid training type')

            if training_type == 'sup':
                loss = decision_maker.do_cross_entropy()
            else:
                loss = decision_maker.do_rl()

            # wandb.log({"batch_loss": loss.item()})
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            decision_maker.clear_buffer()

        callback_args = {'epoch': epoch}
        # wandb.log({'epoch': epoch})
        # wandb.log({'epoch_loss': sum(losses) / len(losses)})
        for callback in callbacks:
            result = callback.execute(**callback_args)
            if result is not None:
                for val, name in result:
                    print(f"{name}: {val}")
                    # wandb.log({name: val})
                    callback_args[name] = val


if __name__ == '__main__':
    print("WARNING: experiment with large waiting time")

    # arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--epochs', type=int, default=100)
    args_parser.add_argument('--batch_size', type=int, default=32)
    args_parser.add_argument('--embedding_size', type=int, default=256)
    args_parser.add_argument('--hidden_size', type=int, default=256)
    args_parser.add_argument('--num_layers', type=int, default=2)
    args_parser.add_argument('--learning_rate', type=float, default=5 * 1e-4)
    args_parser.add_argument('--seed', type=int, default=0)
    args_parser.add_argument('--domain', type=str, default='hoc')


    args = args_parser.parse_args()

    epochs = args.epochs  # 100
    batch_size = args.batch_size  # 32
    embedding_size = args.embedding_size  # 256
    hidden_size = args.hidden_size  # 256
    num_layers = args.num_layers  # 2
    learning_rate = args.learning_rate  # 5 * 10e-4

    dict_size = len(tok2idx)
    seed = args.seed

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    training_type = 'sup'

    add_latent = False
    type_ = args.domain

    save_path = f'models/{type_}/codegen/seed_{seed}/new'

    # config = {
    #     "epochs": epochs,
    #     "batch_size": batch_size,
    #     "dict_size": dict_size,
    #     "embedding_dim": embedding_size,
    #     "hidden_dim": hidden_size,
    #     "num_layers": num_layers,
    #     "learning_rate": learning_rate,
    #     "training_type": training_type,
    #     "save_path": save_path,
    #     "add_latent": add_latent
    # }

    # wandb.login(key=API_KEY)
    # wandb.init(project="intelligent-codegen",
    #            name=f"LSTM-Codegen{nb}-{training_type}",
    #            group=f"Many-to-Many-LSTM-{type_}",
    #            config=config,
    #            mode="online")

    network = LSTMNetwork(dict_size, embedding_size, hidden_size, num_layers, dict_size,
                          add_latent=add_latent, latent_size=64, max_number=16)
    decision_maker = LSTMDecisionMaker(network)

    val_dataset = CodeSpecAndSketchNbDataset(f'data/synthetic/{type_}/seed_{seed}/val.json',
                                             type_=type_)
    train_dataset = CodeSpecAndSketchNbDataset(f'data/synthetic/{type_}/seed_{seed}/train.json',
                                               type_=type_)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, collate_fn=lambda x: x)

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    val_callback = CodeSuccessValCallback(20, decision_maker, val_dataset, type_=type_)
    saving_callback = SavingCallback(decision_maker, save_path, alpha=0.6)

    train(epochs, decision_maker, train_dataloader, optimizer, [val_callback, saving_callback],
          training_type=training_type, type_=type_)
