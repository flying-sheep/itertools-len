"""Test suite."""

from __future__ import annotations

import itertools
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

import pytest

import itertools_len


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from types import ModuleType


abcd = "ABCD"
abcd_combinations = ["AB", "AC", "AD", "BC", "BD", "CD"]


has_pairwise = pytest.mark.skipif(
    not hasattr(itertools, "pairwise"),
    reason="`pairwise` not available",
)


has_batched = pytest.mark.skipif(
    not hasattr(itertools, "pairwise"),
    reason="`batched` not available",
)


def exports(mod: ModuleType) -> None:
    if (explicit := getattr(mod, "__all__", None)) is not None:
        yield from explicit
        return
    for n in dir(mod):
        if not n.startswith("_"):
            yield n


def test_all_available() -> None:
    assert set(exports(itertools)) | {"map"} == set(exports(itertools_len))


@pytest.mark.parametrize(
    ("func_name", "suffix"),
    [
        ("repeat", "endlessly"),
        pytest.param("pairwise", "taken from the input iterator", marks=has_pairwise),
        ("islice", "but returns an iterator"),
    ],
)
def test_wrap_doc(func_name: str, suffix: str) -> None:
    func = getattr(itertools_len, func_name)
    assert f"{suffix}. Wraps" in func.__doc__


@pytest.mark.parametrize("times", [0, 1, 20, None])
def test_repeat(times: int | None) -> None:
    s = itertools_len.repeat("x", times)
    if times is None:
        with pytest.raises(TypeError):
            len(s)
    else:
        assert len(s) == times


@pytest.mark.parametrize(
    ("func_name", "expected", "args"),
    [
        ("accumulate", 2, ([1, 2],)),
        ("starmap", 3, (int, [["1"], ["2"], ["11", 2]])),
        ("map", 3, (int, ["1", "2", "11"])),
        ("zip_longest", 3, ([1, 2, 3], [3])),
    ],
)
def test_maps(func_name: str, expected: int, args: tuple[Any]) -> None:
    func = getattr(itertools_len, func_name)
    res = func(*args)
    assert len(res) == len(list(res)) == expected


@pytest.mark.parametrize(
    ("func", "iterables"),
    [
        (itertools_len.chain, [[1, 2], [3, 4, 5]]),
        (itertools_len.chain.from_iterable, ["abc", "def"]),
    ],
)
def test_chain(
    func: Callable[..., Iterable[Any]],
    iterables: list[Iterable[Any]],
) -> None:
    res = func(*iterables) if func is itertools_len.chain else func(iterables)
    assert len(res) == sum(len(i) for i in iterables)


@has_pairwise
@pytest.mark.parametrize(
    ("n_items", "n_pairs"),
    [(0, 0), (1, 0), (2, 1), (3, 2), (20, 19)],
)
def test_pairwise(n_items: int, n_pairs: int) -> None:
    pairs = itertools_len.pairwise(range(n_items))
    assert len(pairs) == n_pairs


@has_batched
@pytest.mark.parametrize(
    ("n_items", "batch_size", "n_batches"),
    [(0, 3, 0), (1, 3, 1), (2, 2, 1), (3, 2, 2), (20, 3, 7)],
)
def test_batched(n_items: int, batch_size: int, n_batches: int) -> None:
    batches = itertools_len.batched(range(n_items), batch_size)
    assert len(batches) == n_batches


@pytest.mark.parametrize(
    "sl",
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
        # Start, Stop & Step
        slice(0, 1, 2),
        slice(0, 5, 2),
        slice(0, 5, 5),
        # Stop & Step
        slice(0, None, 2),
        slice(0, None, 3),
        # stop > len
        slice(0, 20),
    ],
)
def test_islice(sl: slice) -> None:
    l = list(range(10))
    isl = itertools_len.islice(l, sl.start, sl.stop, sl.step)
    assert len(l[sl]) == len(isl)
    assert l[sl] == list(isl)


def test_islice_kw() -> None:
    l = list(range(10))
    sl = slice(0, 6, 2)
    isl = itertools_len.islice(l, 6, step=2)
    assert len(l[sl]) == len(isl)
    assert l[sl] == list(isl)


@pytest.mark.parametrize("n", range(1, 4))
def test_tee(n: int) -> None:
    r = range(10)
    its = itertools_len.tee(r, n)
    assert len(its) == n
    assert len(its[0]) == 10
    assert next(reversed(its)) == its[-1]
    assert [len(it) for it in its] == [10] * n
    assert [list(it) for it in its] == [list(r) for _ in range(n)]


def test_product_multiple() -> None:
    prod = itertools_len.product("ab", [1, 2])
    assert len(prod) == 4
    assert list(prod) == [("a", 1), ("a", 2), ("b", 1), ("b", 2)]


@pytest.mark.parametrize("reps", [1, 2, 3])
def test_product_rep(reps: int) -> None:
    expected = list(itertools.product(abcd, repeat=reps))
    prod = itertools_len.product(abcd, repeat=reps)
    assert len(expected) == len(prod)
    assert expected == list(prod)


def test_permutations_all() -> None:
    expected = list(itertools.permutations(range(3)))
    perms = itertools_len.permutations(range(3))
    assert len(expected) == len(perms)
    assert expected == list(perms)


@pytest.mark.parametrize("reps", [1, 2, 3])
def test_permutations_subset(reps: int) -> None:
    expected = list(itertools.permutations(abcd, reps))
    perms = itertools_len.permutations(abcd, reps)
    assert len(expected) == len(perms)
    assert expected == list(perms)


@pytest.mark.skipif(not find_spec("scipy"), reason="`scipy` not installed")
@pytest.mark.parametrize(
    ("n", "r", "expected"),
    [
        (len(abcd), 2, len(abcd_combinations)),
    ],
)
def test_ncomb(n: int, r: int, expected: int) -> None:
    assert itertools_len._ncomb_python(n, r) == expected
    assert itertools_len._ncomb_scipy(n, r) == expected


def test_combinations() -> None:
    cmbs = itertools_len.combinations(abcd, 2)
    assert len(cmbs) == len(abcd_combinations)


def test_combinations_combinations_with_replacement() -> None:
    cmbs = itertools_len.combinations_with_replacement(abcd, 2)
    assert len(cmbs) == len(abcd_combinations) + len(abcd)
