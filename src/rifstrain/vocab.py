"""Vocab
=====

This module contains the vocabulary related functions.
For now it only contains a Danish vocabulary.

The module contains the following functions:

    - write_vocab
    - get_vocab
"""

import os
import json


def write_vocab(dir) -> None:
    """Write vocab to file

    Parameters
    ----------
    dir : str
        Directory to write the vocab to.
    """
    with open(os.path.join(dir, "vocab.json"), "w+") as file:
        file.write(json.dumps(get_vocab()))


def get_vocab() -> dict[str, int]:
    """Get vocab from pre-trained model

    Returns
    -------
    vocab : dict
    """
    return {
        "|": 21,
        "'": 13,
        "a": 24,
        "b": 17,
        "c": 25,
        "d": 2,
        "e": 9,
        "f": 14,
        "g": 22,
        "h": 8,
        "i": 4,
        "j": 18,
        "k": 5,
        "l": 16,
        "m": 6,
        "n": 7,
        "o": 10,
        "p": 19,
        "q": 3,
        "r": 20,
        "s": 11,
        "t": 0,
        "u": 26,
        "v": 27,
        "w": 1,
        "x": 23,
        "y": 15,
        "z": 12,
        "æ": 28,
        "ø": 29,
        "å": 30,
        "[UNK]": 31,
        "[PAD]": 32,
    }
