import itertools

import pytest

import itertools_len


def exports(mod):
    for n in dir(mod):
        if not n.startswith("_"):
            yield n


def test_all_available():
    assert set(exports(itertools)) | {"map"} == set(exports(itertools_len))


@pytest.mark.parametrize(
    "slice",
    [
        # Only stop
        slice(None),
        slice(1),
        slice(2),
        slice(3),
        # Start & Stop
        slice(0, 0),
        slice(0, 1),
        slice(0, 2),
        slice(0, 4),
        slice(2, 4),
        slice(2, 8),
        # Start, Stop & step
        slice(0, 1, 2),
        slice(0, 5, 2),
        slice(0, 5, 5),
        # Start & step
        slice(0, None, 2),
        slice(0, None, 3),
        # stop > len
        slice(0, 20),
    ],
)
def test_islice(slice):
    l = list(range(10))
    s = itertools_len.islice(l, slice.start, slice.stop, slice.step)
    assert len(l[slice]) == len(s)
    assert l[slice] == list(s)


def test_product_multiple():
    prod = itertools_len.product("ab", [1, 2])
    assert len(prod) == 4
    assert list(prod) == [("a", 1), ("a", 2), ("b", 1), ("b", 2)]


@pytest.mark.parametrize("reps", [1, 2, 3])
def test_product_rep(reps):
    expected = list(itertools.product("ABCD", repeat=reps))
    prod = itertools_len.product("ABCD", repeat=reps)
    assert len(expected) == len(prod)
    assert expected == list(prod)


def test_permutations_all():
    expected = list(itertools.permutations(range(3)))
    perms = itertools_len.permutations(range(3))
    assert len(expected) == len(perms)
    assert expected == list(perms)


@pytest.mark.parametrize("reps", [1, 2, 3])
def test_permutations_subset(reps):
    expected = list(itertools.permutations("ABCD", reps))
    perms = itertools_len.permutations("ABCD", reps)
    assert len(expected) == len(perms)
    assert expected == list(perms)
