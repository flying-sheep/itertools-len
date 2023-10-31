r"""
Building blocks for iterators, preserving their :func:`len`\ s.

This module contains length-preserving wrappers for all :mod:`itertools`
and the builtin :func:`python:map`. To use it as drop-in replacement, do:

.. code:: python

   import itertools_len as itertools
   from itertools_len import map
"""

from __future__ import annotations

import builtins
import itertools
import sys
from math import ceil
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar


try:
    from typing import ParamSpec
except ImportError:
    from typing_extensions import ParamSpec


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from types import FunctionType


__all__ = []  # Filled below

A = TypeVar("A")
T = TypeVar("T")
C = ParamSpec("C")


class _WrapDocMeta(type):
    _wrapped: FunctionType | type

    @property
    def __doc__(cls) -> str:  # noqa: A003
        # TODO: Allow overriding __doc__
        # https://github.com/flying-sheep/itertools-len/issues/45
        from inspect import getdoc

        patched = "\n".join(
            line
            for line in getdoc(cls._wrapped).splitlines()
            if " --> " not in line and " -> " not in line
        )
        typ = "meth" if "." in cls._wrapped.__qualname__ else "func"
        prefix = "python:" if cls is map else "itertools."
        return f"{patched.strip()} Wraps :{typ}:`{prefix}{cls._wrapped.__qualname__}`."


class _IterTool(metaclass=_WrapDocMeta):
    _wrapped: ClassVar[Callable[C, Iterable[T]]]

    def __init__(self, *args: C.args, **kwargs: C.kwargs) -> None:
        self.itertool = self._wrapped(*args, **kwargs)

    def __iter__(self) -> Iterator[T]:
        return iter(self.itertool)


__all__ += ["count", "cycle", "repeat"]
__doc__ += """
Infinites
---------
:func:`~itertools.count` and :func:`~itertools.cycle` yield infinitely many values
and are therefore simply re-exported. :func:`repeat` is finite if ``times`` is passed.

.. autofunction:: repeat
"""  # noqa: A001


count = itertools.count
cycle = itertools.cycle


class repeat(_IterTool):
    _wrapped = itertools.repeat

    def __init__(self, obj: T, times: int | None = None) -> None:
        super().__init__(obj, *([] if times is None else [times]))
        self.times = times

    def __len__(self) -> int:
        """Calculate how many repetitions are done unless it’s infinite."""
        if self.times is None:
            msg = "Infinite repeat"
            raise TypeError(msg)
        return self.times


__all__ += ["compress", "dropwhile", "filterfalse", "groupby", "takewhile"]
__doc__ += """
Shortening/filtering
--------------------
:func:`~itertools.compress`, :func:`~itertools.dropwhile`,
:func:`~itertools.filterfalse`, :func:`~itertools.groupby`, and
:func:`~itertools.takewhile` all shorten the passed iterable.
Therefore no length can be determined and they are simply re-exported.
"""  # noqa: A001


compress = itertools.compress
dropwhile = itertools.dropwhile
filterfalse = itertools.filterfalse
groupby = itertools.groupby
takewhile = itertools.takewhile


__all__ += ["accumulate", "starmap", "map", "zip_longest"]
__doc__ += """
Mapping
-------
The following functions map an input iterable 1:1 to an output.
For inputs with a length, the output length is the same:

.. autofunction:: accumulate
.. autofunction:: starmap
.. autofunction:: map
.. autofunction:: zip_longest
"""  # noqa: A001


class _IterToolMap(_IterTool):
    def __init__(
        self,
        iterable: Iterable[A],
        *args: C.args,
        **kwargs: C.kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.iterable = iterable

    def __len__(self) -> int:
        """
        Return length of underlying iterable unless it’s no sequence.

        … in which case this will raise an Error.
        """
        return len(self.iterable)


class _Adder:
    """Helper class allowing Sphinx to parse the function signature."""

    def __call__(self, a: A, b: A) -> T:
        a + b

    def __repr__(self) -> str:
        return "add"  # pragma: no cover


class accumulate(_IterToolMap):
    _wrapped = itertools.accumulate

    def __init__(
        self,
        iterable: Iterable[A],
        func: Callable[[A, A], T] = _Adder(),  # noqa: B008
    ) -> None:
        super().__init__(iterable, iterable, func=func)


class starmap(_IterToolMap):
    _wrapped = itertools.starmap

    # Can’t properly type this as Callable[ArgsTuple, T] doesn’t work.
    def __init__(self, function: Callable[..., T], iterable: Iterable[Any]) -> None:
        super().__init__(iterable, function, iterable)


class map(_IterTool):  # noqa: A001
    _wrapped = builtins.map

    # Similar as with starmap
    def __init__(self, func: Callable[..., T], *iterables: Iterable[Any]) -> None:
        super().__init__(func, *iterables)
        self.iterables = iterables

    def __len__(self) -> int:
        """Return length of the shortest iterable."""
        return min(len(iterable) for iterable in self.iterables)


class zip_longest(_IterTool):
    _wrapped = itertools.zip_longest

    def __init__(
        self,
        *iterables: Iterable[Any],
        fillvalue: Any | None = None,  # noqa: ANN401
    ) -> None:
        super().__init__(*iterables, fillvalue=fillvalue)
        self.iterables = iterables

    def __len__(self) -> int:
        """Return length of the longest iterable."""
        return max(len(iterable) for iterable in self.iterables)


__all__ += ["chain"]
__doc__ += """
Chaining
--------
The following functions concatenate the input iterables.
Its length is therefore the sum of the inputs’ lengths.

.. autofunction:: chain
.. autofunction:: itertools_len::chain.from_iterable
"""  # noqa: A001


class _IterToolChain(_IterTool):
    def __init__(
        self,
        iterables: Iterable[Iterable[T]],
        *args: C.args,
        **kw: C.kwargs,
    ) -> None:
        super().__init__(*args, **kw)
        self.iterables = iterables

    def __len__(self) -> int:
        """Sum up the length of chained inputs."""
        # Make sure we don’t iterate over a generator or so
        len(self.iterables)
        return sum(map(len, self.iterables))


class chain(_IterToolChain):
    _wrapped = itertools.chain

    def __init__(self, *iterables: Iterable[T]) -> None:
        super().__init__(iterables, *iterables)

    class from_iterable(_IterToolChain):  # noqa: D106
        _wrapped = itertools.chain.from_iterable

        def __init__(self, iterables: Iterable[Iterable[T]]) -> None:
            super().__init__(iterables, iterables)


if sys.version_info >= (3, 10):
    __all__ += ["pairwise"]
    __doc__ += """
    Pairwise
    --------
    The following function can loop over a sequence in pairs.

    This method has been introduced in Python 3.10, so its length-aware equivalent
    is only available starting from this Python version.

    .. autofunction:: pairwise
    """  # noqa: A001

    class pairwise(_IterTool):
        _wrapped = itertools.pairwise

        def __init__(self, iterable: Iterable[T]) -> None:
            self.iterable = iterable
            super().__init__(self.iterable)

        def __len__(self) -> int:
            """Calculate the number of pairs: max(len-1, 0)."""
            l = len(self.iterable)
            return l - 1 if l > 0 else 0


if sys.version_info >= (3, 12):
    __all__ += ["batched"]
    __doc__ += """
    Batched
    --------
    The following function can loop over a sequence in batches.

    This method has been introduced in Python 3.12, so its length-aware equivalent
    is only available starting from this Python version.

    .. autofunction:: batched
    """  # noqa: A001

    class batched(_IterTool):
        _wrapped = itertools.batched

        def __init__(self, iterable: Iterable[T], n: int) -> None:
            self.iterable = iterable
            self._n = n
            super().__init__(self.iterable, n)

        def __len__(self) -> int:
            """Calculate the number of batches: ceil(len/n)."""
            l = len(self.iterable)
            return ceil(l / self._n)


__all__ += ["islice"]
__doc__ += """
Slicing
-------
The following function slices iterables like :func:`slice`, but lazily.

.. autofunction:: islice
"""  # noqa: A001


class _Missing:
    def __repr__(self) -> str:
        return "missing"  # pragma: no cover


_missing = _Missing()


class islice(_IterTool):
    _wrapped = itertools.islice

    def __init__(
        self,
        iterable: Iterable[T],
        start: int | None,
        stop: int | _Missing = _missing,
        step: int | None = None,
    ) -> None:
        if stop is _missing:
            start, stop = 0, start
        super().__init__(iterable, start, stop, step)
        assert start is None or start >= 0  # noqa: S101
        assert stop is None or stop >= 0  # noqa: S101
        assert step is None or step > 0  # noqa: S101
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
        # else start >= stop
        return 0


__all__ += ["tee"]
__doc__ += """
Splitting
---------
The following function splits an iterable into multiple independent iterators.

.. autofunction:: tee
"""  # noqa: A001


# Can’t subclass _IterTool as we have nothing to be _wrapped.
# Also we initialized with already created itertools.
class _tee:
    def __init__(self, itertool: Iterable[T], it_orig: Iterable[T]) -> None:
        self.itertool = itertool
        self.it_orig = it_orig

    def __iter__(self) -> Iterator[T]:
        return iter(self.itertool)

    def __len__(self) -> int:
        return len(self.it_orig)


# Can’t subclass collections.abc.collection as we already use a metaclass
class tee(metaclass=_WrapDocMeta):
    _wrapped = itertools.tee

    def __init__(self, iterable: Iterable[T], n: int = 2) -> None:
        self.itertools = tuple(_tee(it, iterable) for it in self._wrapped(iterable, n))

    def __getitem__(self, item: int) -> _tee:
        return self.itertools[item]

    def __iter__(self) -> Iterator[_tee]:
        return iter(self.itertools)

    def __reversed__(self) -> Iterator[_tee]:
        return reversed(self.itertools)

    def __len__(self) -> int:
        """Return number of iterators yielded."""
        return len(self.itertools)


__all__ += ["product", "permutations", "combinations", "combinations_with_replacement"]
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
"""  # noqa: A001


class product(_IterTool):
    _wrapped = itertools.product

    def __init__(self, *iterables: Iterable[A], repeat: int = 1) -> None:
        self.sequences = [tuple(i) for i in iterables]
        self.repeat = repeat
        super().__init__(*self.sequences, repeat=repeat)

    def __len__(self) -> int:
        length = 1
        for seq in self.sequences:
            length *= len(seq)
        return length**self.repeat


class permutations(_IterTool):
    _wrapped = itertools.permutations

    def __init__(self, iterable: Iterable, r: int | None = None) -> None:
        self.elements = tuple(iterable)
        self.r = len(self.elements) if r is None else r
        super().__init__(self.elements, r)

    def __len__(self) -> int:
        """Calculate number of r-permutations of n elements [Uspensky37]_."""
        from math import factorial

        n = len(self.elements)
        return factorial(n) // factorial(n - self.r)


def _ncomb_python(n: int, r: int) -> int:
    ncomb = 1
    for i in range(1, min(r, n - r) + 1):
        ncomb *= n
        ncomb //= i
        n -= 1
    return ncomb


def _ncomb_scipy(n: int, r: int) -> int:
    from scipy.special import comb

    return comb(n, r, exact=True)


def _ncomb(n: int, r: int) -> int:
    try:
        return _ncomb_scipy(n, r)
    except ImportError:  # pragma: no cover
        return _ncomb_python(n, r)


class combinations(_IterTool):
    _wrapped = itertools.combinations

    def __init__(self, iterable: Iterable, r: int) -> None:
        self.elements = tuple(iterable)
        self.r = r
        super().__init__(self.elements, r)

    def __len__(self) -> int:
        """Calculate binomial coefficient (n over r)."""
        return _ncomb(len(self.elements), self.r)


class combinations_with_replacement(_IterTool):
    _wrapped = itertools.combinations_with_replacement

    def __init__(self, iterable: Iterable, r: int) -> None:
        self.elements = tuple(iterable)
        self.r = r
        super().__init__(self.elements, r)

    def __len__(self) -> int:
        return _ncomb(self.r + len(self.elements) - 1, self.r)


__doc__ += """
References
----------

.. [Uspensky37] Uspensky et al. (1937),
   *Introduction To Mathematical Probability* p. 18,
   `Mcgraw-hill Book Company London
   <https://archive.org/details/in.ernet.dli.2015.263184/page/n8>`__.
"""  # noqa: A001
