import pytest

import itertools_len


def exports(mod):
    for n in dir(mod):
        if not n.startswith("_"):
            yield n


def test_all_available():
    assert list(exports(itertools_len)) == list(exports(itertools_len))


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
def test_islice_len(slice):
    l = list(range(10))
    assert len(l[slice]) == len(
        itertools_len.islice(l, slice.start, slice.stop, slice.step)
    )
