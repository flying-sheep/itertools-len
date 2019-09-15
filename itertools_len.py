"""
Building blocks for iterators, preserving their ``len()``s.
"""

import itertools
import operator
import typing as t

from get_version import get_version

__version__ = get_version(__file__)
del get_version


class _WrapDocMeta(type):
    @property
    def __doc__(cls):
        # TODO: Allow overriding __doc__
        return cls.__wrapped__.__doc__


class _IterTool(metaclass=_WrapDocMeta):
    __wrapped__: t.Callable

    def __init__(self, *args, **kwargs):
        self.itertool = self.__wrapped__(*args, **kwargs)

    def __iter__(self):
        return iter(self.itertool)


T = t.TypeVar("T")


# Infinites


count = itertools.count
cycle = itertools.cycle


class repeat(_IterTool):
    def __init__(self, object: t.Any, times: t.Optional[int] = None):
        super().__init__(object, times)
        self.times = times

    def __len__(self):
        """Returns how many repetitions are done unless it’s infinite"""
        if self.times is None:
            raise TypeError("Infinite repeat")
        return self.times


# Unknown, shorter


compress = itertools.compress
dropwhile = itertools.combinations
filterfalse = itertools.filterfalse
groupby = itertools.groupby
takewhile = itertools.takewhile


# Maps


class _IterToolMap(_IterTool):
    def __init__(self, iterable: t.Iterable[T], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iterable = iterable

    def __len__(self) -> int:
        """If the underlying iterable is no sequence, this will raise an Error"""
        return len(self.iterable)


class accumulate(_IterToolMap):
    __wrapped__ = itertools.accumulate

    def __init__(
        self, iterable: t.Iterable, func: t.Callable[[T, T], t.Any] = operator.add
    ):
        super().__init__(iterable, iterable, func=func)


# TODO: map


class starmap(_IterToolMap):
    __wrapped__ = itertools.starmap

    # Can’t properly type this as Callable[ArgsTuple, Any] doesn’t work.
    def __init__(self, function: t.Callable, iterable: t.Iterable):
        super().__init__(iterable, function, iterable)


class zip_longest(_IterTool):
    def __init__(self, *iterables: t.Iterable[T], fillvalue: t.Optional[T] = None):
        super().__init__(*iterables, fillvalue=fillvalue)
        self.iterables = iterables

    def __len__(self) -> int:
        """The length of the longest iterable"""
        return max(len(iterable) for iterable in self.iterables)


# Chains


class _IterToolChain(_IterTool):
    def __init__(self, iterables: t.Iterable[t.Iterable], *args):
        super().__init__(*args)
        self.iterables = iterables

    def __len__(self):
        # Make sure we don’t iterate over a generator or so
        len(self.iterables)
        return sum(map(len, self.iterables))


class chain(_IterToolChain):
    __wrapped__ = itertools.chain

    def __init__(self, *iterables: t.Iterable):
        super().__init__(iterables, *iterables)

    class from_iterable(_IterToolChain):
        __wrapped__ = itertools.chain.from_iterable

        def __init__(self, iterables: t.Iterable[t.Iterable]):
            super().__init__(iterables, iterables)


# Slices


class _Missing:
    pass


_missing = _Missing()


class islice(_IterTool):
    __wrapped__ = itertools.islice

    def __init__(
        self,
        iterable: t.Iterable,
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


# Tees


class tee(_IterTool):
    def __init__(self, iterable: t.Iterable, n: int = 2):
        raise NotImplementedError()


# Permutations


class product(_IterTool):
    def __init__(self, *iterables: t.Iterable, repeat: int = 1):
        self.sequences = [tuple(i) for i in iterables]
        self.repeat = repeat
        super().__init__(self.sequences, repeat)

    def __len__(self) -> int:
        length = 1
        for seq in self.sequences:
            length *= len(seq)
        return length * self.repeat


class permutations(_IterTool):
    def __init__(self, iterable: t.Iterable, r: int = None):
        raise NotImplementedError()


class combinations(_IterTool):
    def __init__(self, iterable: t.Iterable, r: int):
        raise NotImplementedError()


class combinations_with_replacement(_IterTool):
    def __init__(self, iterable: t.Iterable, r: int):
        raise NotImplementedError()


# Cleanup


del itertools, operator, t, T
