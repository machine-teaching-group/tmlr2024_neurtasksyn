import json
import os
import random

import numpy as np
import torch

from src.codegen.networks import LSTMNetwork
from src.codegen.utils import tok2idx, idx2tok


class RandomCodeDecisionMaker:
    def __init__(self):
        pass

    def decide(self, root, allowed, lin_code, ):
        return random.choice(allowed)

    def init_self(self):
        pass

    def set_nb_allowed_blocks(self, nb_allowed_blocks):
        pass

    def set_latent(self, latent):
        pass

    def set_tgt_seq(self, tgt_seq):
        pass


class LSTMDecisionMaker:
    def __init__(self, network, tgt_seq=None, latent=None, gamma=0.99):
        self.network = network
        self.tgt_seq = tgt_seq
        self.tgt_seq_idx = 0

        self.buffer = []
        self.last_state = self.network.init_state()

        self.save_to_buffer = True
        self.latent = latent
        self.gamma = gamma

        self.nb_allowed_blocks = None

    def decide(self, root, allowed, lin_code, ):
        if self.latent is not None:
            logits, self.last_state = self.network(torch.tensor([tok2idx[lin_code[-1]]]).unsqueeze(1), self.last_state, self.latent, self.nb_allowed_blocks)
        else:
            logits, self.last_state = self.network(torch.tensor([tok2idx[lin_code[-1]]]).unsqueeze(1), self.last_state, None, self.nb_allowed_blocks)

        # make logits -inf for all non-allowed actions
        for key in tok2idx:
            if key not in allowed:
                logits[0, 0, tok2idx[key]] = -float('inf')

        probs_torch = torch.softmax(logits, dim=-1)
        probs = np.squeeze(probs_torch.tolist())
        probs = probs / np.sum(probs)
        if np.isnan(probs).any() or np.isinf(probs).any():
            print(allowed)
            print("wtf?")
            print(probs)
            print(logits)
            print(probs_torch)
        if self.tgt_seq is not None:
            action = tok2idx[self.tgt_seq[self.tgt_seq_idx]]
            self.tgt_seq_idx += 1
        else:
            action = np.random.choice(len(tok2idx), 1,
                                      p=np.squeeze(probs))[0]

        grad = torch.log(probs_torch[0, 0, action])

        dict_to_add = {
            'action': action,
            'logits': logits,
            'probs': probs_torch,
            'allowed': allowed,
            'grad': grad,
            'ls': None
        }

        if self.tgt_seq is not None:
            expected = self.tgt_seq[self.tgt_seq_idx - 1]
            dict_to_add['tgt'] = tok2idx[expected]
            if expected not in allowed:
                print("wtf?")

        if self.save_to_buffer:
            self.buffer.append(dict_to_add)

        return idx2tok[action]

    def init_self(self):
        self.init_last_state()
        # self.clear_buffer()

    def set_latent(self, latent):
        self.latent = latent

    def get_buffer(self):
        return self.buffer

    def clear_buffer(self):
        self.buffer = []

    def init_last_state(self):
        self.last_state = self.network.init_state()

    def set_tgt_seq(self, tgt_seq):
        self.tgt_seq = tgt_seq
        self.tgt_seq_idx = 0

    def set_nb_allowed_blocks(self, nb_allowed_blocks):
        self.nb_allowed_blocks = nb_allowed_blocks

    def train(self):
        self.save_to_buffer = True
        self.network.train()

    def eval(self):
        self.save_to_buffer = False
        self.network.eval()

    def parameters(self):
        return self.network.parameters()

    def do_cross_entropy(self):
        losses = []
        for i, entry in enumerate(self.buffer):
            logits = entry['logits'].squeeze(1).squeeze(0)
            # mask = torch.zeros_like(logits)
            # for allowed in entry['allowed']:
            #     mask[tok2idx[allowed]] = 1
            loss = torch.nn.functional.cross_entropy(logits,
                                                     torch.tensor(entry['tgt']))
            if loss.isnan().any() or loss.isinf().any():
                print(f"Loss is nan at {i}th entry")
            losses.append(loss)
        return torch.stack(losses).mean()

    def do_end_code(self, lev_dist):
        for idx, entry in enumerate(self.buffer[::-1]):
            if entry["ls"] is None:
                if idx == 0:
                    entry["ls"] = lev_dist
                else:
                    entry["ls"] = self.gamma * self.buffer[-idx]["ls"]
            else:
                break

    def do_rl(self):
        losses = []
        for i, entry in enumerate(self.buffer):
            loss = entry['grad'] * entry['ls']
            if loss.isnan().any() or loss.isinf().any():
                print(f"Loss is nan at {i}th entry")
            losses.append(loss)
        return torch.stack(losses).mean()

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.network.state_dict(), os.path.join(path, "network.pt"))
        metadata = {
            "network_type": self.network.__class__.__name__,
            "dict_size": self.network.dict_size,
            "embedding_size": self.network.embedding_size,
            "output_size": self.network.output_size,
            "lstm_hidden_size": self.network.lstm_hidden_size,
            "nb_layers": self.network.nb_layers,
            "max_number": self.network.max_number,
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        clasz = metadata["network_type"]
        metadata.pop("network_type")
        network = eval(clasz)(**metadata)
        network.load_state_dict(torch.load(os.path.join(path, "network.pt")))
        return cls(network)

if __name__ == '__main__':
    network = LSTMNetwork(10, 10, 10, 10, 10)

    decision_maker = LSTMDecisionMaker(network)

    decision_maker.save('test')

    decision_maker = LSTMDecisionMaker.load('test')
