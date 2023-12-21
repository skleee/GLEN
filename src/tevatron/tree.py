import numpy as np

"""
Referenced from NCI codebase: https://github.com/solidsea98/Neural-Corpus-Indexer-NCI
"""


def decode_token(args, seqs):
    """
    Args:
        seqs: 2d ndarray to be decoded
    Return:
        doc_id string, List[str]
    """
    result = []
    for seq in seqs:
        try:
            eos_idx = seq.tolist().index(1)
            seq = seq[1:eos_idx]
        except:
            print("no eos token found")
        offset = np.arange(len(seq)) * args.output_vocab_size + 2
        res = seq - offset
        result.append("-".join(str(c) for c in res))
    return result


def dec_2d(dec, size):
    """
    Args:
        dec: 1d ndarray
        size: int
    Return:
        List[List[int]]
    """
    res = []
    i = 0
    while i < len(dec):
        res.append(dec[i : i + size])
        i = i + size
    return res


def encode_single_newid(args, seq):
    """
    Args:
        seq: doc_id string to be encoded, like "23456"
    Return:
        List[Int]: encoded tokens
    """
    target_id_int = []
    for i, c in enumerate(seq.split("-")):
        cur_token = i * args.kary + int(c) + 2
        target_id_int.append(cur_token)
    return target_id_int + [1]  # append eos_token


class Node(object):
    def __init__(self, token_id):
        self.token_id = token_id
        self.children = {}

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.token_id) + "\n"
        for child in self.children.values():
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return "<tree node representation>"


class TreeBuilder(object):
    def __init__(self) -> None:
        self.root = Node(0)

    def build(self) -> Node:
        return self.root

    def add(self, seq) -> None:
        """
        seq is List[Int] representing id, without leading pad_token(hardcoded 0), with trailing eos_token(hardcoded 1) and (possible) pad_token, every int is token index
        e.g: [ 9, 14, 27, 38, 47, 58, 62,  1,  0,  0,  0,  0,  0,  0,  0]
        """
        cur = self.root
        for tok in seq:
            if tok == 0:  # reach pad_token
                return
            if tok not in cur.children:
                cur.children[tok] = Node(tok)
            cur = cur.children[tok]
