import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module

from src.emulator.code import Code
from src.emulator.fast_emulator import FastEmulator
from src.symexecution.decision_makers import DecisionMaker
from src.taskgen.feature_processors import get_features_size, \
    get_output_size, lookahead_features_pre_process_input_hoc, \
    dif_features_pre_process_input_hoc, dif_features_pre_process_input_karel, \
    dif_features_pre_process_input_karel_compact, grid_and_coverage_pre_process_input_hoc, \
    grid_coverage_code_type_pre_process_input_hoc, grid_coverage_latent_pre_process_input_hoc, \
    grid_and_coverage_pre_process_input_karel, grid_and_coverage_pre_process_input_karel_nomarkernumber, \
    grid_and_coverage_pre_process_input_karel_markerbitmap, grid_and_coverage_pre_process_input_karel_extended, \
    grid_and_coverage_pre_process_input_karel_binary_markers_for_filtered, \
    grid_and_coverage_pre_process_input_karel_for_filtered, \
    next_grid_only_and_coverage_pre_process_input_hoc, grid_and_coverage_action_count_pre_process_input_hoc, \
    grid_and_coverage_action_count_pre_process_input_karel, \
    grid_and_coverage_action_count_pre_process_input_karel_markerbitmap, \
    grid_and_coverage_action_count_pre_process_input_compact_karel_markerbitmap
from src.taskgen.indexed_code import IndexedCode
from src.taskgen.networks import HandmadeFeaturesNetwork, ActionValueHandmadeFeaturesNetwork, \
    GridActionValueHandmadeFeaturesNetwork, ResActionValueNetwork
from src.taskgen.vocabularies import decision2idx, \
    idx2decision


class IntelligentDecisionMaker(DecisionMaker):
    def __init__(self,
                 network: Module,
                 preprocess_function: str,
                 emulator: Optional[FastEmulator] = None,
                 has_buffer: bool = False,
                 gamma: float = 0.9999,
                 ):
        self._emulator = emulator

        self.buffer = []
        self.network = network

        self.has_buffer = has_buffer
        self.current_example = None
        self.current_example_index = -1

        self.preprocess_function = eval(preprocess_function)
        self.gamma = gamma

    def set_emulator(self, emulator):
        self._emulator = emulator

    def process_code(self, code: Code):
        return code

    def set_current_example(self, example):
        self.current_example = example
        self.current_example_index = -1

    def binary_decision(self):
        inp = self.preprocess_function(
            self._emulator
        )
        logits = self.network(inp)
        logits[[value for key, value in decision2idx.items()
                if 'binary' not in key]] = float("-inf")
        probabilities = F.softmax(logits)
        if self.current_example:
            self.current_example_index += 1
            decision = self.current_example[self.current_example_index]
        else:
            decision = torch.multinomial(probabilities, 1).item()
        if self.has_buffer:
            to_mem = {
                "symworld": self._emulator.state.world.to_json(),
                "actions": self._emulator.state.actions,
                "ticks": self._emulator.state.ticks,
                "current": self._emulator.current_location,
                "code": self._emulator.ast,
                "decision": f'binary:{decision}',
                "probabilities": probabilities,
                "reward": None,
            }

            self.buffer.append(to_mem)

        return int(idx2decision[decision].split(":")[1])

    def pick_int(self, from_, to, for_):
        inp = self.preprocess_function(
            self._emulator
        )
        logits = self.network(inp)
        logits[[value for key, value in decision2idx.items()
                if for_ not in key]] = float("-inf")
        probabilities = F.softmax(logits)
        if self.current_example:
            self.current_example_index += 1
            decision = self.current_example[self.current_example_index]
        else:
            decision = torch.multinomial(probabilities, 1).item()
        if self.has_buffer:
            to_mem = {
                "symworld": self._emulator.state.world,
                "actions": self._emulator.state.actions,
                "ticks": self._emulator.state.ticks,
                "current": self._emulator.current_location,
                "decision": f'{for_}:{decision}',
                "probabilities": probabilities,
                "reward": None,
            }

            self.buffer.append(to_mem)

        return int(idx2decision[decision].split(":")[1])

    def reset_buffer(self):
        self.buffer = []

    def populate_rewards(self, reward):
        for idx, entry in enumerate(self.buffer[::-1]):
            if entry["reward"] is None:
                if idx == 0:
                    entry["reward"] = reward
                else:
                    entry["reward"] = self.gamma * self.buffer[-idx]["reward"]
            else:
                break

    def compute_loss(self):
        pass

    def train(self):
        self.network.train()
        self.has_buffer = True

    def eval(self):
        self.network.eval()
        self.current_example = None
        self.current_example_index = -1
        self.has_buffer = False

    def get_parameters(self):
        return self.network.parameters()

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.network.state_dict(), os.path.join(path, "network.pt"))
        metadata = {
            "agent_type": self.__class__.__name__,
            "preprocess_function": self.preprocess_function.__name__,
            "network": self.network.__class__.__name__,
            "layer_sizes": self.network.layer_sizes,
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        network = eval(metadata["network"])(
            get_features_size(metadata['preprocess_function']),
            metadata["layer_sizes"],
            get_output_size())
        network.load_state_dict(torch.load(os.path.join(path, "network.pt")))
        return cls(network, metadata["preprocess_function"])

    def __deepcopy__(self, memodict={}):
        copy_ = IntelligentDecisionMaker(self.network,
                                         self.preprocess_function.__name__,
                                         self._emulator,
                                         self.has_buffer)
        copy_.buffer = self.buffer
        copy_.current_example = self.current_example
        copy_.current_example_index = self.current_example_index
        return copy_


class GridActorCriticLookaheadDecisionMaker(IntelligentDecisionMaker):
    def __init__(self,
                 network: Module,
                 preprocess_function: str,
                 emulator: Optional[FastEmulator] = None,
                 has_buffer: bool = False,
                 gamma: float = 0.9999,
                 temperature=1.0,
                 entropy=False
                 ):
        self._emulator = emulator

        self.buffer = []
        self.network = network

        assert isinstance(self.network, GridActionValueHandmadeFeaturesNetwork) or \
               isinstance(self.network, ResActionValueNetwork)
        assert preprocess_function in ['grid_and_coverage_pre_process_input_hoc',
                                       'grid_coverage_code_type_pre_process_input_hoc',
                                       'grid_coverage_latent_pre_process_input_hoc',
                                       'grid_and_coverage_pre_process_input_karel',
                                       'grid_and_coverage_pre_process_input_karel_nomarkernumber',
                                       'grid_and_coverage_pre_process_input_karel_markerbitmap',
                                       'grid_and_coverage_pre_process_input_karel_extended',
                                       'grid_and_coverage_pre_process_input_karel_for_filtered',
                                       'grid_and_coverage_pre_process_input_karel_binary_markers_for_filtered',
                                       'next_grid_only_and_coverage_pre_process_input_hoc',
                                       'grid_and_coverage_action_count_pre_process_input_hoc',
                                       'grid_and_coverage_action_count_pre_process_input_karel',
                                       'grid_and_coverage_action_count_pre_process_input_karel_markerbitmap',
                                       'grid_and_coverage_action_count_pre_process_input_compact_karel_markerbitmap']

        self.has_buffer = has_buffer
        self.current_example = None
        self.current_example_index = -1

        self.preprocess_function = eval(preprocess_function)
        self.preprocess_function_name = preprocess_function
        self.gamma = gamma

        self.orig_temperature = temperature
        self.temperature = temperature
        self.entropy = entropy

        self.code_type = None

    def set_code_type(self, code_type):
        self.code_type = code_type

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.network.state_dict(), os.path.join(path, "network.pt"))
        dct = self.network.get_to_save()
        metadata = {
            "agent_type": self.__class__.__name__,
            "preprocess_function": self.preprocess_function.__name__,
            "network": self.network.__class__.__name__,
            "network_params": dct
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        network = eval(metadata["network"])(**metadata["network_params"])
        state_dict = torch.load(os.path.join(path, "network.pt"))
        network.load_state_dict(state_dict)
        return cls(network, metadata['preprocess_function'])

    def process_code(self, code: Code):
        return IndexedCode(code)

    def train(self):
        self.network.train()
        self.temperature = self.orig_temperature
        self.has_buffer = True

    def eval(self):
        self.network.eval()
        self.current_example = None
        self.current_example_index = -1
        self.has_buffer = False
        self.temperature = 1.0

    def binary_decision(self):
        self.network.eval()

        possible_decisions = [dec for dec in decision2idx if 'binary' in dec]

        heur = []
        for dec in possible_decisions:
            if self.preprocess_function_name in ['grid_coverage_code_type_pre_process_input_hoc',
                                                 'grid_coverage_latent_pre_process_input_hoc']:
                heur.append(self.preprocess_function(dec,
                                                     self._emulator,
                                                     self.code_type))
            else:
                heur.append(self.preprocess_function(dec,
                                                     self._emulator))
        act, val = self.network(heur)

        self.network.train()

        act = act / self.temperature

        probabilities = F.softmax(act, dim=0)
        if self.entropy:
            # add small value to avoid log(0)
            probabilities = probabilities + 1e-8
            # normalize to add back to 1
            probabilities = probabilities / probabilities.sum()
            entropy = -(probabilities * torch.log(probabilities)).sum()
            if entropy.isnan().any():
                print(probabilities)
        decision = np.random.choice(len(possible_decisions), 1,
                                    p=np.squeeze(probabilities.detach().numpy()))[0]
        if self.has_buffer:
            grad_val = torch.log(probabilities.squeeze(0)[decision])
            val = val.squeeze(0)[decision]
            raw_score = act.squeeze(0).detach().numpy()[decision]
            to_mem = {
                "symworld": self._emulator.state.world,
                "actions": self._emulator.state.actions,
                "ticks": self._emulator.state.ticks,
                "current": self._emulator.current_location,
                "decision": f'binary:{decision}',
                "grad": grad_val,
                "reward": None,
                "value": val,
            }

            self.buffer.append(to_mem)

        return decision

    def pick_int(self, from_, to, for_):
        self.network.eval()

        possible_decisions = [dec for dec in decision2idx if for_ in dec]
        heur = []
        for dec in possible_decisions:
            if self.preprocess_function_name in ['grid_coverage_code_type_pre_process_input_hoc',
                                                 'grid_coverage_latent_pre_process_input_hoc']:
                heur.append(self.preprocess_function(dec,
                                                     self._emulator,
                                                     self.code_type))
            else:
                heur.append(self.preprocess_function(dec,
                                                     self._emulator))

        # print("obtained heuristics")

        act, val = self.network(heur)

        self.network.train()

        probabilities = F.softmax(act, dim=0)
        if self.entropy:
            # add small value to avoid log(0)
            probabilities = probabilities + 1e-8
            # normalize to add back to 1
            probabilities = probabilities / probabilities.sum()
            entropy = -(probabilities * torch.log(probabilities)).sum()
            if entropy.isnan().any():
                print(probabilities)
        decision = np.random.choice(len(possible_decisions), 1,
                                    p=np.squeeze(probabilities.detach().numpy()))[0]

        if self.has_buffer:
            grad_val = torch.log(probabilities.squeeze(0)[decision])
            val = val.squeeze(0)[decision]
            raw_score = act.squeeze(0).detach().numpy()[decision]
            to_mem = {
                "symworld": self._emulator.state.world,
                "actions": self._emulator.state.actions,
                "ticks": self._emulator.state.ticks,
                "current": self._emulator.current_location,
                "decision": f'binary:{decision}',
                "grad": grad_val,
                "reward": None,
                "value": val,
            }

            if self.entropy:
                to_mem["entropy"] = entropy

            self.buffer.append(to_mem)

        decision = int(possible_decisions[decision].split(':')[-1])

        return decision

    def compute_loss(self):
        discounted_rewards = [x['reward'] for x in self.buffer]

        discounted_rewards = torch.tensor(discounted_rewards)

        policy_gradient = []
        value_loss = []
        for b, ret in zip(self.buffer, discounted_rewards):
            pol_loss = -b['grad'] * (ret - b['value'].item())
            policy_gradient.append(pol_loss)
            value_loss.append(F.smooth_l1_loss(b['value'], torch.tensor([ret])))

        policy_gradient = torch.stack(policy_gradient).sum()
        value_loss = torch.stack(value_loss).sum()

        if discounted_rewards.mean() > 1:
            print(discounted_rewards.mean())

        if self.entropy:
            entropy = torch.stack([x['entropy'] for x in self.buffer]).sum()
            return policy_gradient + value_loss, entropy, discounted_rewards.detach().numpy()

        return policy_gradient + value_loss, discounted_rewards.detach().numpy()

    def get_parameters(self):
        return self.network.parameters()

    def __deepcopy__(self, memodict={}):
        # print("deepcopying")
        # net_copy = copy.deepcopy(self.network)
        # print("deepcopied")
        copy_ = GridActorCriticLookaheadDecisionMaker(self.network,
                                                      self.preprocess_function.__name__,
                                                      self._emulator,
                                                      self.has_buffer,
                                                      self.gamma,
                                                      self.temperature,
                                                      self.entropy)
        copy_.buffer = self.buffer
        copy_.current_example = self.current_example
        copy_.current_example_index = self.current_example_index
        return copy_
