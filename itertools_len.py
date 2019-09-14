import itertools
import operator
import typing as t


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
repeat = itertools.repeat


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
    def __init__(self, iterables: t.Collection[t.Iterable], *args):
        super().__init__(*args)
        self.iterables = iterables

    def __len__(self):
        return sum(map(len, self.iterables))


class chain(_IterToolChain):
    __wrapped__ = itertools.chain

    def __init__(self, *iterables):
        super().__init__(iterables, *iterables)

    class from_iterable(_IterToolChain):
        __wrapped__ = itertools.chain.from_iterable

        def __init__(self, iterables):
            super().__init__(iterables, iterables)


# Slices


class islice(_IterTool):
    def __init__(
        self,
        iterable: t.Iterable,
        start: int,
        stop: t.Optional[int] = None,
        step: t.Optional[int] = None,
    ):
        if stop is None and step is None:
            start, stop = 0, start
        raise NotImplementedError()


# Tees


class tee(_IterTool):
    def __init__(self, iterable: t.Iterable, n: int = 2):
        raise NotImplementedError()


# Permutations


class product(_IterTool):
    def __init__(self, *iterables: t.Iterable, repeat: int = 1):
        raise NotImplementedError()


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
