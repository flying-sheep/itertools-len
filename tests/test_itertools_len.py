import itertools

import pytest

import itertools_len


has_pairwise = pytest.mark.skipif(
    not hasattr(itertools, "pairwise"), reason="`pairwise` not available"
)


def exports(mod):
    if (explicit := getattr(mod, "__all__", None)) is not None:
        yield from explicit
        return
    for n in dir(mod):
        if not n.startswith("_"):
            yield n


def test_all_available():
    assert set(exports(itertools)) | {"map"} == set(exports(itertools_len))


@pytest.mark.parametrize(
    "func_name,suffix",
    [
        ("repeat", "endlessly"),
        pytest.param("pairwise", "taken from the input iterator", marks=has_pairwise),
        ("islice", "but returns an iterator"),
    ],
)
def test_wrap_doc(func_name, suffix):
    func = getattr(itertools_len, func_name)
    assert f"{suffix}. Wraps" in func.__doc__


@pytest.mark.parametrize("times", [0, 1, 20, None])
def test_repeat(times):
    if times is None:
        with pytest.raises(TypeError):
            itertools_len.repeat("x", times)
    else:
        s = itertools_len.repeat("x", times)
        assert len(s) == times


@pytest.mark.parametrize(
    "func_name,expected,args",
    [
        ("accumulate", 2, ([1, 2],)),
        ("starmap", 3, (int, [["1"], ["2"], ["11", 2]])),
        ("map", 3, (int, ["1", "2", "11"])),
        ("zip_longest", 3, ([1, 2, 3], [3])),
    ],
)
def test_maps(func_name, expected, args):
    func = getattr(itertools_len, func_name)
    res = func(*args)
    assert len(res) == len(list(res)) == expected


@pytest.mark.parametrize(
    "func,iterables",
    [
        (itertools_len.chain, [[1, 2], [3, 4, 5]]),
        (itertools_len.chain.from_iterable, ["abc", "def"]),
    ],
)
def test_chain(func, iterables):
    res = func(*iterables) if func is itertools_len.chain else func(iterables)
    assert len(res) == sum(len(i) for i in iterables)


@has_pairwise
@pytest.mark.parametrize("n_items,n_pairs", [(0, 0), (1, 0), (2, 1), (3, 2), (20, 19)])
def test_pairwise(n_items, n_pairs):
    pairs = itertools_len.pairwise(range(n_items))
    assert len(pairs) == n_pairs


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
def test_islice(sl):
    l = list(range(10))
    isl = itertools_len.islice(l, sl.start, sl.stop, sl.step)
    assert len(l[sl]) == len(isl)
    assert l[sl] == list(isl)


def test_islice_kw():
    l = list(range(10))
    sl = slice(0, 6, 2)
    isl = itertools_len.islice(l, 6, step=2)
    assert len(l[sl]) == len(isl)
    assert l[sl] == list(isl)


@pytest.mark.parametrize("n", range(1, 4))
def test_tee(n):
    r = range(10)
    its = itertools_len.tee(r, n)
    assert len(its) == n
    assert len(its[0]) == 10
    assert list(reversed(its))[0] == its[-1]
    assert [len(it) for it in its] == [10] * n
    assert [list(it) for it in its] == [list(r) for _ in range(n)]


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


abcd = "ABCD"
abcd_combinations = ["AB", "AC", "AD", "BC", "BD", "CD"]


def test_combinations():
    cmbs = itertools_len.combinations(abcd, 2)
    assert len(cmbs) == len(abcd_combinations)


def test_combinations_combinations_with_replacement():
    cmbs = itertools_len.combinations_with_replacement(abcd, 2)
    assert len(cmbs) == len(abcd_combinations) + len(abcd)
