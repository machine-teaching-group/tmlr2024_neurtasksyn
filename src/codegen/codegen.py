from itertools import combinations, chain

from src.codegen.codegraph import json_to_ast, ast_to_json_karelgym_format
from src.codegen.symast import json_to_symast, SymAst, symast_to_json
from src.codegen.utils import translate_condition

symbolic2concrete_hoc = {
    "repeatUntil": ["repeat_until_goal(bool_goal)"],
    "repeat": ["repeat(2)", "repeat(3)", "repeat(4)", "repeat(5)", "repeat(6)", "repeat(7)", "repeat(8)", "repeat(9)",
               "repeat(10)"],
    "if": ["if(bool_path_ahead)", "if(bool_path_left)", "if(bool_path_right)"],
    "ifelse": ["ifelse(bool_path_ahead)", "ifelse(bool_path_left)", "ifelse(bool_path_right)"]
}

symbolic2concrete_karel = {
    'while': ['while(bool_path_ahead)',
              'while(bool_no_path_ahead)',
              'while(bool_marker_present)',
              'while(bool_no_marker_present)'],
    'if': ['if(bool_path_ahead)',
           'if(bool_path_left)',
           'if(bool_path_right)',
           'if(bool_no_path_ahead)',
           'if(bool_marker_present)',
           'if(bool_no_marker_present)'],
    'ifelse': ['ifelse(bool_path_ahead)',
               'ifelse(bool_path_left)',
               'ifelse(bool_path_right)',
               'ifelse(bool_no_path_ahead)',
               'ifelse(bool_marker_present)',
               'ifelse(bool_no_marker_present)'],
    'repeat': ['repeat(2)', 'repeat(3)', 'repeat(4)', 'repeat(5)', 'repeat(6)',
               'repeat(7)', 'repeat(8)', 'repeat(9)', 'repeat(10)'],
}


def compute_available_blocks(ast, num_blocks_allowed):
    """
    Computes the number of blocks available for a given ast.
    """
    if num_blocks_allowed < 0:
        raise ValueError('Number of blocks allowed exceeded.')

    for child in ast.children():
        if child._type in ['D', 'repeatUntil', 'repeat', 'ifelse', 'if', 'while'] or '(' in child._type:
            num_blocks_allowed -= 1
            num_blocks_allowed = compute_available_blocks(child, num_blocks_allowed)
        elif child._type in ['do', 'else']:
            num_blocks_allowed = compute_available_blocks(child, num_blocks_allowed)
        elif child._type != 'A':
            num_blocks_allowed -= 1
        elif child._type == 'A' and len(ast.children()) == 1:
            num_blocks_allowed -= 1
        else:
            pass

    return num_blocks_allowed


def create_allowed_for_A(parent, current_index, num_blocks_allowed, type_='hoc'):
    if num_blocks_allowed['num'] <= 0 and len(parent.children()) > 1:
        basic_actions = []
    else:
        if type_ == 'hoc':
            basic_actions = ['move', 'turn_left', 'turn_right']
        elif type_ == 'karel':
            basic_actions = ['move', 'turn_left', 'turn_right', 'pick_marker', 'put_marker']
    if current_index == len(parent.children()) - 1:
        if current_index > 0:
            return basic_actions + [']']
        return basic_actions
    else:
        next_child = parent.children()[current_index + 1]
        additionals = []
        if next_child._type in ['D', 'repeatUntil', 'repeat', 'if', 'ifelse', 'while'] \
                or hasattr(next_child, '_condition') or hasattr(next_child, '_values'):
            if hasattr(next_child, '_values') and len(next_child._values) > 0:
                x = [next_child._values[0]]
            elif hasattr(next_child, '_condition') and next_child._condition != None:
                x = [next_child._type + '(' + next_child._condition + ')']
            else:
                x = next_child._type
                if type_ == 'hoc':
                    x = symbolic2concrete_hoc[x]
                elif type_ == 'karel':
                    x = symbolic2concrete_karel[x]

            additionals += x
        return basic_actions + additionals


def decide_for_node(root: SymAst, parent: SymAst, current_index: int,
                    decision_maker, lin_code, taken_decision: dict, num_blocks_allowed: dict,
                    type_='hoc'):
    if root._type == 'run':
        decision = decision_maker.decide(root, ['run'], lin_code)
    elif root._type == 'A':
        allowed = create_allowed_for_A(parent, current_index, num_blocks_allowed, type_)
        decision = decision_maker.decide(root, allowed,
                                         lin_code)
    else:
        if taken_decision['decision'] is not None:
            decision = taken_decision['decision']
            taken_decision['decision'] = None
        else:
            decision = decision_maker.decide(root, [root._type], lin_code)

        if root._type == 'repeatUntil':
            root._values = [decision]
            root._type = 'D'
        elif root._type == 'ifelse':
            root._values = [decision]
            root._type = 'D'
        elif root._type == 'if':
            root._values = [decision]
            root._type = 'D'
        elif root._type == 'repeat':
            root._values = [decision]
            root._type = 'D'
        elif root._type == 'while':
            root._values = [decision]
            root._type = 'D'

    return decision


def populate_body(root: SymAst,
                  parent: SymAst,
                  current_index: int,
                  decision_maker,
                  num_blocks_allowed: dict,
                  lin_code: list,
                  taken_decision: dict,
                  type_='hoc'):
    if root._type in ['repeatUntil', 'repeat', 'run', 'while']:
        lin_code.append(
            decide_for_node(root, parent, current_index, decision_maker, lin_code, taken_decision, num_blocks_allowed,
                            type_))
        lin_code.append(decision_maker.decide(root, ['['], lin_code))
        idx = 0
        while idx < len(root.children()):
            child = root.children()[idx]
            populate_body(child, root, idx, decision_maker, num_blocks_allowed, lin_code, taken_decision, type_)
            idx += 1
            if taken_decision['decision'] not in [None, ']']:
                idx -= 1
            elif taken_decision['decision'] == ']':
                lin_code.append(taken_decision['decision'])
                taken_decision['decision'] = None
        if len(root.children()[-1]._values) > 0 and root.children()[-1]._values[0] == 'repeat_until_goal(bool_goal)':
            lin_code.append(decision_maker.decide(root, [']'], lin_code))
    elif 'ifelse' in root._type:
        lin_code.append(
            decide_for_node(root, parent, current_index, decision_maker, lin_code, taken_decision, num_blocks_allowed,
                            type_))
        lin_code.append(decision_maker.decide(root, ['['], lin_code))
        for child in root.children():
            if child._type == 'do':
                lin_code.append(decision_maker.decide(root, ['do'], lin_code))
                lin_code.append(decision_maker.decide(root, ['['], lin_code))
                for idx, c in enumerate(child.children()):
                    populate_body(c, child, idx, decision_maker, num_blocks_allowed, lin_code, taken_decision, type_)
                lin_code.append(taken_decision['decision'])
                taken_decision['decision'] = None
            elif child._type == 'else':
                lin_code.append(decision_maker.decide(root, ['else'], lin_code))
                lin_code.append(decision_maker.decide(root, ['['], lin_code))
                for idx, c in enumerate(child.children()):
                    populate_body(c, child, idx, decision_maker, num_blocks_allowed, lin_code, taken_decision, type_)
                lin_code.append(taken_decision['decision'])
                taken_decision['decision'] = None
        lin_code.append(decision_maker.decide(root, [']'], lin_code))
    elif 'if' in root._type:
        lin_code.append(
            decide_for_node(root, parent, current_index, decision_maker, lin_code, taken_decision, num_blocks_allowed,
                            type_))
        lin_code.append(decision_maker.decide(root, ['['], lin_code))
        for child in root.children():
            if child._type == 'do':
                lin_code.append(decision_maker.decide(root, ['do'], lin_code))
                lin_code.append(decision_maker.decide(root, ['['], lin_code))
                for idx, c in enumerate(child.children()):
                    populate_body(c, child, idx, decision_maker, num_blocks_allowed, lin_code, taken_decision, type_)
                lin_code.append(taken_decision['decision'])
                taken_decision['decision'] = None
        lin_code.append(decision_maker.decide(root, [']'], lin_code))
    elif '(' in root._type or (hasattr(root, '_condition') and root._condition is not None) or \
            (hasattr(root, '_values') and len(root._values) > 0 and '(' in root._values[0]):
        lin_code.append(
            decide_for_node(root, parent, current_index, decision_maker, lin_code, taken_decision, num_blocks_allowed,
                            type_))
        if 'if' in root._type or (hasattr(root, '_values') and len(root._values) > 0 and 'if' in root._values[0]):
            lin_code.append(decision_maker.decide(root, ['['], lin_code))
            for child in root.children():
                if child._type == 'do':
                    lin_code.append(decision_maker.decide(root, ['do'], lin_code))
                    lin_code.append(decision_maker.decide(root, ['['], lin_code))
                    for idx, c in enumerate(child.children()):
                        populate_body(c, child, idx, decision_maker, num_blocks_allowed, lin_code, taken_decision,
                                      type_)
                    lin_code.append(taken_decision['decision'])
                    taken_decision['decision'] = None
                elif child._type == 'else':
                    lin_code.append(decision_maker.decide(root, ['else'], lin_code))
                    lin_code.append(decision_maker.decide(root, ['['], lin_code))
                    for idx, c in enumerate(child.children()):
                        populate_body(c, child, idx, decision_maker, num_blocks_allowed, lin_code, taken_decision,
                                      type_)
                    lin_code.append(taken_decision['decision'])
                    taken_decision['decision'] = None
            lin_code.append(decision_maker.decide(root, [']'], lin_code))
        else:
            lin_code.append(decision_maker.decide(root, ['['], lin_code))
            idx = 0
            while idx < len(root.children()):
                child = root.children()[idx]
                populate_body(child, root, idx, decision_maker, num_blocks_allowed, lin_code, taken_decision, type_)
                idx += 1
                if taken_decision['decision'] not in [None, ']']:
                    idx -= 1
                elif taken_decision['decision'] == ']':
                    lin_code.append(taken_decision['decision'])
                    taken_decision['decision'] = None
                if len(root.children()[-1]._values) > 0 and root.children()[-1]._values[
                    0] == 'repeat_until_goal(bool_goal)':
                    lin_code.append(decision_maker.decide(root, [']'], lin_code))
    elif root._type == 'A':
        decision = decide_for_node(root, parent, current_index, decision_maker, lin_code, taken_decision,
                                   num_blocks_allowed, type_)
        if decision not in ['move', 'turn_left', 'turn_right', 'pick_marker', 'put_marker']:
            taken_decision['decision'] = decision
            parent.children().remove(root)
        else:
            if len(parent.children()) > 1:
                num_blocks_allowed['num'] -= 1
                decision_maker.set_nb_allowed_blocks(num_blocks_allowed['num'])
            parent.children().insert(current_index, SymAst(decision, 'ACTION_BLK'))
            lin_code.append(decision)
    elif root._type in ['move', 'turn_left', 'turn_right', 'pick_marker', 'put_marker']:
        lin_code.append(root._type)

    return root, lin_code


def sketch2code(sketch_json: dict,
                decision_maker,
                num_blocks_allowed: int,
                tgt=None,
                latent=None,
                type_='hoc'):
    """
    Converts a sketch into a code.
    """

    # Convert the sketch into a SymAst.
    if not isinstance(sketch_json, SymAst):
        sketch_ast = json_to_symast(sketch_json)
    else:
        sketch_ast = sketch_json

    decision_maker.init_self()
    decision_maker.set_tgt_seq(tgt)
    if latent is not None:
        decision_maker.set_latent(latent)

    num_blocks_allowed = compute_available_blocks(sketch_ast, num_blocks_allowed)
    if num_blocks_allowed < 0:
        raise ValueError('Not enough blocks to generate the code.')
    decision_maker.set_nb_allowed_blocks(num_blocks_allowed)

    lin_code = ['<start>']
    taken_decision = {"decision": None}
    code_ast, lin_code = populate_body(sketch_ast, None, 0, decision_maker, {'num': num_blocks_allowed},
                                       lin_code=lin_code,
                                       taken_decision=taken_decision,
                                       type_=type_)

    symast_json = symast_to_json(code_ast)

    code_ast = json_to_ast(symast_json)
    code_karelgym_format = ast_to_json_karelgym_format(code_ast)

    return code_karelgym_format, lin_code[1:]

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def get_control_structures_combinations_rec(sketch_ast, num_combinations):
    """
    Returns the number of combinations of control structures.
    """
    if sketch_ast._type in ['repeatUntil', 'repeat', 'ifelse', 'if', 'while']:
        num_combinations += [sketch_ast._type + '_' + sketch_ast._id]
    for child in sketch_ast.children():
        if child._type in ['repeatUntil', 'repeat', 'ifelse', 'if', 'do', 'else', 'while']:
            num_combinations = get_control_structures_combinations_rec(child, num_combinations)

    return num_combinations


def get_control_structures_combinations(sketch_ast, half=False):
    """
    Returns the number of combinations of control structures.
    """
    num_combinations = []
    num_combinations = get_control_structures_combinations_rec(sketch_ast, num_combinations)

    if not half:
        return [x for x in powerset(num_combinations)]

    lngth = max([len(x) for x in powerset(num_combinations)])
    return [x for x in powerset(num_combinations) if len(x) == lngth // 2]


def get_partial_code_from_sketch_code_comb(sketch_ast, code_dct, comb):
    """
    Returns the partial code from the sketch and the code.
    """
    if sketch_ast._type in ['repeatUntil', 'repeat', 'ifelse', 'if', 'while']:
        if sketch_ast._id == None:
            print('here')
        if sketch_ast._type + '_' + sketch_ast._id in comb:
            sketch_ast._type = translate_condition[code_dct['type']][
                code_dct['condition'] if 'condition' in code_dct else code_dct['times']]
            sketch_ast._values = [translate_condition[code_dct['type']][
                                      code_dct['condition'] if 'condition' in code_dct else code_dct['times']]]
    code_dct_index = 0
    for child in sketch_ast.children():
        if child._type in ['repeatUntil', 'repeat', 'ifelse', 'if', 'while']:
            get_partial_code_from_sketch_code_comb(child, code_dct['body'][code_dct_index], comb)
            code_dct_index += 1
        if sketch_ast._type == 'else':
            for idx in range(code_dct_index, len(code_dct['elseBody'])):
                if code_dct['elseBody'][idx]['type'] in ['move', 'turnRight', 'turnLeft', 'pickMarker', 'putMarker']:
                    code_dct_index += 1
                else:
                    break
        elif sketch_ast._type == 'do':
            if 'body' in code_dct:
                q = 'body'
            else:
                q = 'ifBody'
            for idx in range(code_dct_index, len(code_dct[q])):
                if code_dct[q][idx]['type'] in ['move', 'turnRight', 'turnLeft', 'pickMarker', 'putMarker']:
                    code_dct_index += 1
                else:
                    break
        elif child._type == 'A':
            if 'body' not in code_dct:
                print('here')
            for idx in range(code_dct_index, len(code_dct['body'])):
                if code_dct['body'][idx]['type'] in ['move', 'turnRight', 'turnLeft', 'pickMarker', 'putMarker']:
                    code_dct_index += 1
                else:
                    break

    return sketch_ast