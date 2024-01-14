import re

rollouts_lookup = {
    'hoc':
        {
            0: {
                'rollouts': 100,
                'patience': 50,
            },
            1: {
                'rollouts': 100,
                'patience': 50,
            },
            2: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            3: {
                'rollouts': 100,
                'patience': 50,
            },
            4: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            5: {
                'rollouts': 100,
                'patience': 50,
            },
            6: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            7: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            8: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            9: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            10: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            11: {
                'rollouts': 100,
                'patience': 50,
            },
            12: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            13: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            14: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            15: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            16: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            17: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            18: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            19: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            20: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            21: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            22: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            23: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            24: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            25: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            26: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            27: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            28: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            29: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            30: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            31: {
                'rollouts': 100,
                'patience': 50,
            },
            32: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            33: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            34: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            35: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            36: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            37: {
                'rollouts': 100,
                'patience': 50,
            },
            38: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            39: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            40: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
        },
    'karel':
        {
            0: {
                'rollouts': 100,
                'patience': 50,
            },
            1: {
                'rollouts': 100,
                'patience': 50,
            },
            2: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            3: {
                'rollouts': 100,
                'patience': 50,
            },
            4: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            5: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            6: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            7: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            8: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            9: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            10: {
                'rollouts': 100,
                'patience': 50,
            },
            11: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            12: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            13: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
            14: {
                'rollouts': 1_000_000,
                'patience': 250_000,
            },
        }
}

def extract_number(f):
    s = re.findall("\d+$", f)
    return int(s[0]) if s else -1, f