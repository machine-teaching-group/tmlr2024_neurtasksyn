import json

from torch.utils.data import Dataset


def get_code_type(code: dict):
    if_present = False
    repeat_until_present = False
    for x in code['body']:
        if x['type'] in ['if', 'ifElse']:
            if_present = True
        elif x['type'] == 'repeatUntil' or x['type'] == 'while':
            repeat_until_present = True
            break
        elif x['type'] == 'repeat':
            if_pres_rec, repeat_until_pres_rec = get_code_type(x)
            if if_pres_rec:
                if_present = True
            if repeat_until_pres_rec:
                repeat_until_present = True

    return if_present, repeat_until_present


def get_code_type_wrapper(code: dict):
    if_present, repeat_until_present = get_code_type(code)
    if repeat_until_present:
        return 2
    elif if_present:
        return 1
    else:
        return 0


class CodeDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path

        self.data = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                code = json.loads(line)
                self.data.append((code, get_code_type_wrapper(code['program_json']), code['score'] if 'score' in code else 0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class DifficultCodeDataset(Dataset):
    def __init__(self, file_path, sk_nb):
        self.file_path = file_path

        self.data = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                code = json.loads(line)
                if code['sketch_nb'] >= sk_nb:
                    self.data.append((code, get_code_type_wrapper(code['program_json']), code['score'] if 'score' in code else 0))
                # self.data.append((code, get_code_type_wrapper(code['program_json'])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]