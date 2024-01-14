import copy
import json
import os

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from src.codegen.code_sketches import KAREL_SKETCHES, CODE_SKETCHES
from src.codegen.codegen import get_control_structures_combinations, get_partial_code_from_sketch_code_comb
from src.codegen.converter import karelcode_json_to_ast
from src.codegen.symast import SymAst
from src.codegen.utils import linearize_ast


class CodeSpecAndSketchNbDataset(Dataset):
    def __init__(self, file_path, type_='hoc'):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                code = json.loads(line)
                number = code['sketch_nb']
                if type_ == 'hoc':
                    sk = CODE_SKETCHES[number][0]
                else:
                    sk = KAREL_SKETCHES[number][0]
                code_type = SymAst.from_json(sk)
                lst = get_control_structures_combinations(code_type)
                code_dict = code['program_json']
                tgt = karelcode_json_to_ast(code_dict)
                tgt = linearize_ast(tgt)

                for comb in lst:
                    partial_sk = get_partial_code_from_sketch_code_comb(copy.deepcopy(code_type), code_dict,
                                                                        comb)
                    # num_blocks_allowed = compute_available_blocks(partial_sk, 15)
                    self.data.append((partial_sk, number, tgt, line, None))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]