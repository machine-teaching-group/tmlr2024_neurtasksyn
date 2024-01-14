import json

from src.emulator.code import Code


class IndexedCode(Code):
    def __init__(self, code: Code):
        super().__init__(code.type, code.astJson)
        self._index_ast(code.astJson)

    def _index_ast(self, ast_json):
        self._index_block(ast_json["body"], 0)

    def _index_block(self, block_json, index):
        for block in block_json:
            block["index"] = index
            index += 1
            if block["type"] == "repeat":
                index = self._index_block(block["body"], index)
            elif block["type"] == "while":
                index = self._index_block(block["body"], index)
            elif block["type"] == "if":
                index = self._index_block(block["body"], index)
            elif block["type"] == "ifElse":
                index = self._index_block(block["ifBody"], index)
                index = self._index_block(block["elseBody"], index)
            elif block["type"] == "repeatUntil":
                index = self._index_block(block["body"], index)
        return index
