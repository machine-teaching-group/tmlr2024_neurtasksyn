class SymAst:

    # assumes that values, val_const will be null
    @staticmethod
    def from_json(json_dict):
        return json_to_symast(json_dict)

    def to_json(self):
        return symast_to_json(self)

    def __init__(self, node_type_str, id, val=[], children=[], minval=None):
        self._type = node_type_str
        self._id = id  # str: required for tracking the number of constructs (D/A) in the code
        self._values = val
        self._minval = minval  # size constraint on the node values
        self._children = children
        self._size = 0
        self._depth = 0

        # counts of the constructs
        self._n_D = 0
        self._n_A = 0

        for child in children:
            self._size += child._size
            self._depth = max(self._depth, child._depth)
            self._n_D += child._n_D
            self._n_A += child._n_A

        if self._type in ['if', 'ifelse', 'repeatUntil', 'repeat', 'while', 'D']:
            self._size += 1
            self._depth += 1
            self._n_D += 1
        elif self._type in ['A', 'move', 'turn_left', 'turn_right', 'pick_marker', 'put_marker', 'X']:  # do not update size in this case
            self._n_A += 1
        elif self._type == 'run':  # empty code is of size = 0; depth = 1
            self._depth += 1
        else:
            self._size += 1

    def size(self):
        return self._size

    def depth(self):
        return self._depth

    def children(self):
        return self._children

    def label_print(self):
        if self._id:
            label = self._type + '_' + self._id + ' vals:' + str(self._values)
        else:
            label = self._type + ' vals:' + str(self._values)
        return label

    def __repr__(self, offset=''):
        cs = offset + self.label_print() + '\n'
        for child in self.children():
            cs += offset + child.__repr__(offset + '   ')
        return cs


def json_to_symast(root: dict):
    ''' Converts a JSON dictionary to SymAST Node'''

    def get_children(json_node):
        children = json_node.get('children', [])
        return children

    if root['type'] not in ['turn_left', 'turn_right', 'move', 'pick_marker', 'put_marker'] and '(' not in root['type']:
        split = root['type'].split('_')
        node_type = split[0]
        if len(split) > 1:
            node_id = root['type'].split('_')[1]
        else:
            node_id = None
        children = get_children(root)
        minval = root.get('minval')
        values = root.get('values')
        if values is None:
            val = []
        else:
            val = values
    elif '(' in root['type']:
        node_type = root['type']
        node_id = None
        children = get_children(root)
        minval = root.get('minval')
        values = [root['type']]
        if values is None:
            val = []
        else:
            val = values
    else:
        node_type = root['type']
        node_id = None
        children = get_children(root)
        minval = None
        val = []

    if node_type == 'run':
        return SymAst('run', 'RUN', val, [json_to_symast(child) for child in children],
                      minval=None)
    elif node_type in ['if', 'ifelse', 'repeatUntil', 'repeat', 'while', 'D', 'do', 'else']:
        return SymAst(node_type, node_id, val, [json_to_symast(child) for child in children],
                      minval=minval)
    elif node_type in ['A', 'move', 'turn_left', 'turn_right', 'pick_marker', 'put_marker', 'X']:
        return SymAst(node_type, node_id, val, [json_to_symast(child) for child in children],
                      minval=minval)
    elif '(' in node_type:
        return SymAst(node_type, node_id, val, [json_to_symast(child) for child in children],
                      minval=minval)
    else:
        assert "Unknown node type encountered!"
        return None


def symast_to_json(root: SymAst):
    ''' Converts an ASTNode into a JSON dictionary.'''

    if len(root.children()) == 0:
        if root._type == 'D':
            if root._values:
                node_dict = {'type': root._values[0]}
            else:
                node_dict = {'type': root._type + '_' + root._id}
        else:
            if root._type == 'A':
                node_dict = {'type': root._type + '_' + root._id}
            else:
                node_dict = {'type': root._type}
        return node_dict

    if root._type == 'D':
        if root._values:
            node_dict = {'type': root._values[0]}
        else:
            node_dict = {'type': root._type + '_' + root._id}
    else:
        if root._type == 'A':
            node_dict = {'type': root._type + '_' + root._id}
        else:
            node_dict = {'type': root._type}

    children = [symast_to_json(child) for child in root.children()]

    if children:
        node_dict['children'] = children

    return node_dict


if __name__ == '__main__':
    basic_code_type = {'type': 'run',
                       'children': [
                           {'type': 'A_1'},
                           {'type': 'repeatUntil_1',
                            'children': [
                                {'type': 'A_2'},
                                {'type': 'ifelse_1',
                                 'children': [
                                     {'type': 'do_0',
                                      'children': [
                                          {'type': 'A_3'},
                                      ]},
                                     {'type': 'else_0',
                                      'children': [
                                          {'type': 'A_4'},
                                      ]}
                                 ]},
                                {'type': 'A_5'}
                            ]},
                           {'type': 'A_6'}
                       ]}

    ast = SymAst.from_json(basic_code_type)
    print(ast)
