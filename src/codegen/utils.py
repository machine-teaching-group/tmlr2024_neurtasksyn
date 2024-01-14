import json
from pprint import pprint

from src.codegen.codegraph import ASTNode
from src.codegen.converter import karelcode_json_to_ast
from src.codegen.symast import json_to_symast

OTHER_HOC_BLOCKS = ['repeat_until_goal(bool_goal)',
                    'repeat(2)', 'repeat(3)', 'repeat(4)',
                    'repeat(5)', 'repeat(6)', 'repeat(7)',
                    'repeat(8)', 'repeat(9)', 'repeat(10)',
                    'if(bool_path_ahead)',
                    'if(bool_path_left)',
                    'if(bool_path_right)',
                    'ifelse(bool_path_ahead)',
                    'ifelse(bool_path_left)',
                    'ifelse(bool_path_right)'
                    ]

REPEAT_BLOCKS = ['repeat(2)', 'repeat(3)', 'repeat(4)',
                 'repeat(5)', 'repeat(6)', 'repeat(7)',
                 'repeat(8)', 'repeat(9)', 'repeat(10)']

IF_HOC_BLOCKS = ['if(bool_path_ahead)',  # 'if(bool_no_path_ahead)',
                 'if(bool_path_left)',  # 'if(bool_no_path_left)',
                 'if(bool_path_right)']  # 'if(bool_no_path_right)']

IFELSE_HOC_BLOCKS = ['ifelse(bool_path_ahead)',  # 'ifelse(bool_no_path_ahead)',
                     'ifelse(bool_path_left)',  # 'ifelse(bool_no_path_left)',
                     'ifelse(bool_path_right)']  # 'ifelse(bool_no_path_right)']

REPEAT_UNTIL_BLOCKS = ['repeat_until_goal(bool_goal)']

BASIC_HOC_ACTION_BLOCKS = ['move', 'turn_left', 'turn_right']

tok2idx = {
    "pad": 0,
    "move": 1,
    "turn_left": 2,
    "turn_right": 3,
    "repeat_until_goal(bool_goal)": 4,
    "repeat(2)": 5,
    "repeat(3)": 6,
    "repeat(4)": 7,
    "repeat(5)": 8,
    "repeat(6)": 9,
    "repeat(7)": 10,
    "repeat(8)": 11,
    "repeat(9)": 12,
    "repeat(10)": 13,
    "if(bool_path_ahead)": 14,
    "if(bool_path_left)": 15,
    "if(bool_path_right)": 16,
    "ifelse(bool_path_ahead)": 17,
    "ifelse(bool_path_left)": 18,
    "ifelse(bool_path_right)": 19,
    "do": 20,
    "else": 21,

    "repeatUntil": 22,
    "repeat": 23,
    "if": 24,
    "ifelse": 25,
    "run": 26,
    "A": 27,
    "[": 28,
    "]": 29,

    "X": 30,
    "repeatUntil(X)": 31,
    "repeat(X)": 32,
    "if(X)": 33,
    "ifelse(X)": 34,

    "<start>": 35,
    "<end>": 36,

    "while(bool_path_ahead)": 37,
    "while(bool_path_left)": 38,
    "while(bool_path_right)": 39,
    "while(bool_marker_present)": 40,
    "while(bool_no_marker_present)": 41,
    "while(bool_no_path_right)": 42,
    "while(bool_no_path_left)": 43,
    "while(bool_no_path_ahead)": 44,
    "while": 45,
    "while(X)": 46,
    "if(bool_marker_present)": 47,
    "if(bool_no_marker_present)": 48,
    "if(bool_no_path_right)": 49,
    "if(bool_no_path_left)": 50,
    "if(bool_no_path_ahead)": 51,
    "ifelse(bool_marker_present)": 52,
    "ifelse(bool_no_marker_present)": 53,
    "ifelse(bool_no_path_right)": 54,
    "ifelse(bool_no_path_left)": 55,
    "ifelse(bool_no_path_ahead)": 56,
    "put_marker": 57,
    "pick_marker": 58,
}

translate_condition = {
    "repeatUntil": {"boolGoal": "repeat_until_goal(bool_goal)"},
    "repeat": {2: "repeat(2)", 3: "repeat(3)", 4: "repeat(4)", 5: "repeat(5)", 6: "repeat(6)",
               7: "repeat(7)", 8: "repeat(8)", 9: "repeat(9)", 10: "repeat(10)"},
    "if": {"frontIsClear": "if(bool_path_ahead)", "leftIsClear": "if(bool_path_left)",
           "rightIsClear": "if(bool_path_right)", "markersPresent": "if(bool_marker_present)",
           "noMarkersPresent": "if(bool_no_marker_present)", "notRightIsClear": "if(bool_no_path_right)",
           "notLeftIsClear": "if(bool_no_path_left)", "notFrontIsClear": "if(bool_no_path_ahead)"},
    "ifElse": {"frontIsClear": "ifelse(bool_path_ahead)", "leftIsClear": "ifelse(bool_path_left)",
               "rightIsClear": "ifelse(bool_path_right)", "markersPresent": "ifelse(bool_marker_present)",
               "noMarkersPresent": "ifelse(bool_no_marker_present)", "notRightIsClear": "ifelse(bool_no_path_right)",
               "notLeftIsClear": "ifelse(bool_no_path_left)", "notFrontIsClear": "ifelse(bool_no_path_ahead)"},
    'while': {"frontIsClear": "while(bool_path_ahead)", "leftIsClear": "while(bool_path_left)",
              "rightIsClear": "while(bool_path_right)", "markersPresent": "while(bool_marker_present)",
              "noMarkersPresent": "while(bool_no_marker_present)", "notRightIsClear": "while(bool_no_path_right)",
              "notLeftIsClear": "while(bool_no_path_left)", "notFrontIsClear": "while(bool_no_path_ahead)"}
}

idx2tok = {v: k for k, v in tok2idx.items()}


def find_next_action(reference_root: ASTNode, partial_dict):
    """
    Finds the next action in the reference tree.
    """
    if not isinstance(partial_dict, dict):
        partial_dict = partial_dict.to_json()

    for index, child in enumerate(partial_dict['children']):
        if 'A' in child['type']:
            if index > len(reference_root.children()) - 1:
                return 'endBody', index
            node = reference_root.children()[index]
            if node._type not in ['move', 'turn_left', 'turn_right']:
                return 'endBody', index
            return node._type, index
        elif '(' in child['type'] or child['type'] in ['do', 'else']:
            rtn = find_next_action(reference_root.children()[index], child)
            if rtn is not None:
                return rtn
        elif 'repeatUntil' in child['type'] or 'ifelse' in child['type'] or 'if' in child['type'] or 'repeat' in child[
            'type']:
            node = reference_root.children()[index]
            return node._type + '(' + node._condition + ')', index

    return None


def get_action_token_for_node(node: ASTNode):
    """
    Returns the action token for the given node.
    """
    if node._type == 'D' or hasattr(node, '_condition'):
        if hasattr(node, '_values') and len(node._values) > 0:
            x = node._values[0]
        elif hasattr(node, '_condition') and node._condition != None:
            x = node._type + '(' + node._condition + ')'
        else:
            x = node._type
        if x == 'r':
            print("here")
        return x
    return node._type


def linearize_ast_old(root: ASTNode):
    """
    Linearizes the AST into a list of nodes.
    """
    nodes = [get_action_token_for_node(root)]
    if root._type in ['repeatUntil', 'repeat', 'run']:
        for child in root.children():
            nodes.extend(linearize_ast_old(child))
        nodes.append(']')
    elif 'ifelse' in root._type:
        for child in root.children():
            if child._type == 'do':
                nodes.append('do')
                for c in child.children():
                    nodes.extend(linearize_ast_old(c))
                nodes.append(']')
            elif child._type == 'else':
                nodes.append('else')
                for c in child.children():
                    nodes.extend(linearize_ast_old(c))
                nodes.append(']')
    elif 'if' in root._type:
        for child in root.children():
            if child._type == 'do':
                nodes.append('do')
                for c in child.children():
                    nodes.extend(linearize_ast_old(c))
                nodes.append(']')
    elif '(' in root._type or (hasattr(root, '_condition') and root._condition is not None) or \
            (hasattr(root, '_values') and len(root._values) > 0 and '(' in root._values[0]):
        if 'if' in root._type or (hasattr(root, '_values') and len(root._values) > 0 and 'if' in root._values[0]):
            for child in root.children():
                if child._type == 'do':
                    nodes.append('do')
                    for c in child.children():
                        nodes.extend(linearize_ast_old(c))
                    nodes.append(']')
                elif child._type == 'else':
                    nodes.append('else')
                    for c in child.children():
                        nodes.extend(linearize_ast_old(c))
                    nodes.append(']')
        else:
            for child in root.children():
                nodes.extend(linearize_ast_old(child))
            nodes.append(']')
    # elif root._type == 'D':
    #     print("here")
    return nodes


def linearize_ast(root: ASTNode):
    """
    Linearizes the AST into a list of nodes.
    """
    nodes = [get_action_token_for_node(root)]
    if root._type in ['repeatUntil', 'repeat', 'run']:
        nodes.append('[')
        for child in root.children():
            nodes.extend(linearize_ast(child))
        nodes.append(']')
    elif 'ifelse' in root._type:
        nodes.append('[')
        for child in root.children():
            if child._type == 'do':
                nodes.append('do')
                nodes.append('[')
                for c in child.children():
                    nodes.extend(linearize_ast(c))
                nodes.append(']')
            elif child._type == 'else':
                nodes.append('else')
                nodes.append('[')
                for c in child.children():
                    nodes.extend(linearize_ast(c))
                nodes.append(']')
        nodes.append(']')
    elif 'if' in root._type:
        nodes.append('[')
        for child in root.children():
            if child._type == 'do':
                nodes.append('do')
                nodes.append('[')
                for c in child.children():
                    nodes.extend(linearize_ast(c))
                nodes.append(']')
        nodes.append(']')
    elif '(' in root._type or (hasattr(root, '_condition') and root._condition is not None) or \
            (hasattr(root, '_values') and len(root._values) > 0 and '(' in root._values[0]):
        if 'if' in root._type or (hasattr(root, '_values') and len(root._values) > 0 and 'if' in root._values[0]):
            nodes.append('[')
            for child in root.children():
                if child._type == 'do':
                    nodes.append('do')
                    nodes.append('[')
                    for c in child.children():
                        nodes.extend(linearize_ast(c))
                    nodes.append(']')
                elif child._type == 'else':
                    nodes.append('else')
                    nodes.append('[')
                    for c in child.children():
                        nodes.extend(linearize_ast(c))
                    nodes.append(']')
            nodes.append(']')
        else:
            nodes.append('[')
            for child in root.children():
                nodes.extend(linearize_ast(child))
            nodes.append(']')
    # elif root._type == 'D':
    #     print("here")
    return nodes


def convert2idx(lst):
    return [tok2idx[tok] for tok in lst]


if __name__ == '__main__':
    code_json = '''{"type": "run", "body": [{"type": "move"}, {"type": "turnRight"}, {"type": 
    "repeatUntil", "condition": "boolGoal", "body": [{"type": "move"}, {"type": "ifElse", "condition": 
    "rightIsClear", "ifBody": [{"type": "turnRight"}, {"type": "turnRight"}], "elseBody": [{"type": "move"}, 
    {"type": "move"}, {"type": "turnRight"}]}]}]}'''

    sketch_json = {'type': 'run',
                   'children': [
                       {'type': 'move'},
                       {'type': 'turn_right'},
                       {'type': 'A_3'},
                       {'type': 'repeat_until_goal(bool_goal)',
                        'children': [
                            {'type': 'move'},
                            {'type': 'ifelse(bool_path_right)',
                             'children': [
                                 {'type': 'do',
                                  'children': [
                                      {'type': 'turn_right'},
                                      {'type': 'turn_right'},
                                      {'type': 'A_3'},
                                  ]},
                                 {'type': 'else',
                                  'children': [
                                      {'type': 'move'},
                                      {'type': 'move'},
                                      {'type': 'turn_right'},
                                      {'type': 'A_4'},
                                  ]}
                             ]},
                            # {'type': 'A_5'}
                        ]},
                       # {'type': 'A_6'}
                   ]}

    code_json = json.loads(code_json)
    ast = karelcode_json_to_ast(code_json)

    sketch_ast = json_to_symast(sketch_json)

    # sketch_ast = sketch_ast.to_json()
    # pprint(sketch_ast)

    # print(ast)
    # print(sketch_ast)

    # partial_dict = find_next_action(ast, sketch_ast)
    # print(partial_dict)

    print(linearize_ast(sketch_ast))
    print(convert2idx(linearize_ast(sketch_ast)))
