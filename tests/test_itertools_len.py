import inspect
import itertools

import pytest

import itertools_len


def exports(mod):
    for n in dir(mod):
        if not n.startswith("_"):
            yield n


def test_all_available():
    assert list(exports(itertools_len)) == list(exports(itertools_len))
