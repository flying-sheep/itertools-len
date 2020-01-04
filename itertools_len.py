r"""
Building blocks for iterators, preserving their :func:`len`\ s.

This module contains length-preserving wrappers for all :mod:`itertools`
and the builtin :func:`map`. To use it as drop-in replacement, do:

.. code:: python

   import itertools_len as itertools
   from itertools_len import map
"""

import itertools
import builtins
import operator
import typing as t
from types import FunctionType

from get_version import get_version

__version__ = get_version(__file__)

A = t.TypeVar("A")
T = t.TypeVar("T")


class _WrapDocMeta(type):
    _wrapped: t.Union[FunctionType, t.Type]

    @property
    def __doc__(cls) -> str:
        # TODO: Allow overriding __doc__
        from inspect import getdoc

        patched = "\n".join(
            line for line in getdoc(cls._wrapped).splitlines() if ") --> " not in line
        )
        patched = patched.replace("repeat(object [,times]) -> ", "")
        typ = "meth" if "." in cls._wrapped.__qualname__ else "func"
        prefix = "" if cls is map else "itertools."
        return f"{patched.strip()} Wraps :{typ}:`{prefix}{cls._wrapped.__qualname__}`."


class _IterTool(metaclass=_WrapDocMeta):
    _wrapped: t.ClassVar[t.Callable]

    def __init__(self, *args, **kwargs):
        self.itertool = self._wrapped(*args, **kwargs)

    def __iter__(self) -> t.Iterator[T]:
        return iter(self.itertool)


__doc__ += """
Infinites
---------
:func:`~itertools.count` and :func:`~itertools.cycle` yield infinitely many values
and are therefore simply re-exported. :func:`repeat` is finite if ``times`` is passed.

.. autofunction:: repeat
"""


count = itertools.count
cycle = itertools.cycle


class repeat(_IterTool):
    _wrapped = itertools.repeat

    def __init__(self, object: T, times: t.Optional[int] = None):
        super().__init__(object, times)
        self.times = times

    def __len__(self) -> int:
        """Returns how many repetitions are done unless it’s infinite"""
        if self.times is None:
            raise TypeError("Infinite repeat")
        return self.times


__doc__ += """
Shortening/filtering
--------------------
:func:`~itertools.compress`, :func:`~itertools.combinations`,
:func:`~itertools.filterfalse`, :func:`~itertools.groupby`, and
:func:`~itertools.takewhile` all shorten the passed iterable.
Therefore no length can be determined and they are simply re-exported.
"""


compress = itertools.compress
dropwhile = itertools.combinations
filterfalse = itertools.filterfalse
groupby = itertools.groupby
takewhile = itertools.takewhile


__doc__ += """
Mapping
-------
The following functions map an input iterable 1:1 to an output.
For inputs with a length, the output length is the same:

.. autofunction:: accumulate
.. autofunction:: starmap
.. autofunction:: map
.. autofunction:: zip_longest
"""


class _IterToolMap(_IterTool):
    def __init__(self, iterable: t.Iterable[A], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iterable = iterable

    def __len__(self) -> int:
        """If the underlying iterable is no sequence, this will raise an Error"""
        return len(self.iterable)


class accumulate(_IterToolMap):
    _wrapped = itertools.accumulate

    def __init__(
        self, iterable: t.Iterable[A], func: t.Callable[[A, A], T] = operator.add
    ):
        super().__init__(iterable, iterable, func=func)


class starmap(_IterToolMap):
    _wrapped = itertools.starmap

    # Can’t properly type this as Callable[ArgsTuple, T] doesn’t work.
    def __init__(self, function: t.Callable[..., T], iterable: t.Iterable[t.Any]):
        super().__init__(iterable, function, iterable)


class map(_IterTool):
    _wrapped = builtins.map

    # Similar as with starmap
    def __init__(self, func: t.Callable[..., T], *iterables: t.Iterable[t.Any]):
        super().__init__(func, *iterables)
        self.iterables = iterables

    def __len__(self) -> int:
        """The length of the shortest iterable"""
        return min(len(iterable) for iterable in self.iterables)


class zip_longest(_IterTool):
    _wrapped = itertools.zip_longest

    def __init__(
        self, *iterables: t.Iterable[t.Any], fillvalue: t.Optional[t.Any] = None
    ):
        super().__init__(*iterables, fillvalue=fillvalue)
        self.iterables = iterables

    def __len__(self) -> int:
        """The length of the longest iterable"""
        return max(len(iterable) for iterable in self.iterables)


__doc__ += """
Chaining
--------
The following functions concatenate the input iterables.
Its length is therefore the sum of the inputs’ lengths.

.. autofunction:: chain
.. autofunction:: itertools_len::chain.from_iterable
"""


class _IterToolChain(_IterTool):
    def __init__(self, iterables: t.Iterable[t.Iterable[T]], *args):
        super().__init__(*args)
        self.iterables = iterables

    def __len__(self) -> int:
        # Make sure we don’t iterate over a generator or so
        len(self.iterables)
        return sum(map(len, self.iterables))


class chain(_IterToolChain):
    _wrapped = itertools.chain

    def __init__(self, *iterables: t.Iterable[T]):
        super().__init__(iterables, *iterables)

    class from_iterable(_IterToolChain):
        _wrapped = itertools.chain.from_iterable

        def __init__(self, iterables: t.Iterable[t.Iterable[T]]):
            super().__init__(iterables, iterables)


__doc__ += """
Slicing
-------
The following function slices iterables like :func:`slice`, but lazily.

.. autofunction:: islice
"""


class _Missing:
    pass


_missing = _Missing()


class islice(_IterTool):
    _wrapped = itertools.islice

    def __init__(
        self,
        iterable: t.Iterable[T],
        start: t.Optional[int],
        stop: t.Union[int, _Missing] = _missing,
        step: t.Optional[int] = None,
    ):
        if stop is _missing:
            start, stop = 0, start
        super().__init__(iterable, start, stop, step)
        assert start is None or start >= 0
        assert stop is None or stop >= 0
        assert step is None or step > 0
        self.iterable = iterable
        self.start = 0 if start is None else start
        self.stop = stop
        self.step = 1 if step is None else step

    def __len__(self) -> int:
        stop = self.stop
        if stop is None or stop > len(self.iterable):
            stop = len(self.iterable)
        import math

        if self.start < stop:
            return math.ceil((stop - self.start) / self.step)
        else:  # start >= stop
            return 0


__doc__ += """
Splitting
---------
The following function splits an iterable into multiple independent iterators.

.. autofunction:: tee
"""


# Can’t subclass _IterTool as we have nothing to be _wrapped.
# Also we initialized with already created itertools.
class _tee:
    def __init__(self, itertool: t.Iterable[T], it_orig: t.Iterable[T]):
        self.itertool = itertool
        self.it_orig = it_orig

    def __iter__(self) -> t.Iterator[T]:
        return iter(self.itertool)

    def __len__(self) -> int:
        return len(self.it_orig)


# Can’t subclass collections.abc.collection as we already use a metaclass
class tee(metaclass=_WrapDocMeta):
    _wrapped = itertools.tee

    def __init__(self, iterable: t.Iterable[T], n: int = 2):
        self.itertools = tuple(_tee(it, iterable) for it in self._wrapped(iterable, n))

    def __getitem__(self, item: int) -> _tee:
        return self.itertools[item]

    def __iter__(self) -> t.Iterator[_tee]:
        return iter(self.itertools)

    def __reversed__(self) -> t.Iterator[_tee]:
        return reversed(self.itertools)

    def len(self) -> int:
        """Number of iterators returned"""
        return len(self.itertools)


__doc__ += """
Permutations and combinations
-----------------------------
The following functions return permutations and combinations of input sequences.

.. autofunction:: product
.. autofunction:: permutations
.. automethod:: permutations.__len__
.. autofunction:: combinations
.. automethod:: combinations.__len__
.. autofunction:: combinations_with_replacement
"""


class product(_IterTool):
    _wrapped = itertools.product

    def __init__(self, *iterables: t.Iterable[A], repeat: int = 1):
        self.sequences = [tuple(i) for i in iterables]
        self.repeat = repeat
        super().__init__(*self.sequences, repeat=repeat)

    def __len__(self) -> int:
        length = 1
        for seq in self.sequences:
            length *= len(seq)
        return length ** self.repeat


class permutations(_IterTool):
    _wrapped = itertools.permutations

    def __init__(self, iterable: t.Iterable, r: int = None):
        self.elements = tuple(iterable)
        self.r = len(self.elements) if r is None else r
        super().__init__(self.elements, r)

    def __len__(self) -> int:
        """
        The number of r-permutations of n elements [Uspensky37]_.

        .. [Uspensky37] Uspensky et al. (1937),
           *Introduction To Mathematical Probability* p. 18,
           `Mcgraw-hill Book Company London
           <https://archive.org/details/in.ernet.dli.2015.263184/page/n8>`.
        """
        from math import factorial

        n = len(self.elements)
        return factorial(n) // factorial(n - self.r)


def _ncomb(n: int, r: int) -> int:
    try:
        from scipy.special import comb
    except ImportError:
        pass
    else:
        return comb(n, r, exact=True)

    ncomb = 1
    for i in range(1, min(r, n - r) + 1):
        ncomb *= n
        ncomb //= i
        n -= 1
    return ncomb


class combinations(_IterTool):
    _wrapped = itertools.combinations

    def __init__(self, iterable: t.Iterable, r: int):
        self.elements = tuple(iterable)
        self.r = r
        super().__init__(self.elements, r)

    def __len__(self) -> int:
        """The binomial coefficient (n over r)"""
        return _ncomb(len(self.elements), self.r)


class combinations_with_replacement(_IterTool):
    _wrapped = itertools.combinations_with_replacement

    def __init__(self, iterable: t.Iterable, r: int):
        self.elements = tuple(iterable)
        self.r = r
        super().__init__(self.elements, r)

    def __len__(self) -> int:
        return _ncomb(self.r + len(self.elements) - 1, self.r)


# Cleanup


del builtins, itertools, operator, t, FunctionType, get_version, A, T
