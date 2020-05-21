import weakref
from typing import Callable, Generic, Iterable, Iterator, List, Optional, Sequence, TypeVar, overload

__all__ = [
    "chunk",
    "take",
    "drop",
    "drop_until",
    "split_by",
    "scanl",
    "scanr",
    "LazyList",
    "Range",
    "MapList",
]

T = TypeVar('T')
A = TypeVar('A')
B = TypeVar('B')
R = TypeVar('R')


def chunk(n: int, iterable: Iterable[T]) -> Iterator[List[T]]:
    r"""Split the iterable into chunks, with each chunk containing no more than ``n`` elements.

    .. code:: python

        >>> list(chunk(3, range(10)))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    :param n: The maximum number of elements in one chunk.
    :param iterable: The iterable.
    :return: An iterator over chunks.
    """
    if n <= 0:
        raise ValueError("`n` should be positive")
    group = []
    for x in iterable:
        group.append(x)
        if len(group) == n:
            yield group
            group = []
    if len(group) > 0:
        yield group


def take(n: int, iterable: Iterable[T]) -> Iterator[T]:
    r"""Take the first :attr:`n` elements from an iterable.

    .. code:: python

        >>> list(take(5, range(1000000)))
        [0, 1, 2, 3, 4]

    :param n: The number of elements to take.
    :param iterable: The iterable.
    :return: An iterator returning the first :attr:`n` elements from the iterable.
    """
    if n < 0:
        raise ValueError("`n` should be non-negative")
    try:
        it = iter(iterable)
        for _ in range(n):
            yield next(it)
    except StopIteration:
        pass


def drop(n: int, iterable: Iterable[T]) -> Iterator[T]:
    r"""Drop the first :attr:`n` elements from an iterable, and return the rest as an iterator.

    .. code:: python

        >>> next(drop(5, range(1000000)))
        5

    :param n: The number of elements to drop.
    :param iterable: The iterable.
    :return: An iterator returning the remaining part of the iterable after the first :attr:`n` elements.
    """
    if n < 0:
        raise ValueError("`n` should be non-negative")
    try:
        it = iter(iterable)
        for _ in range(n):
            next(it)
        yield from it
    except StopIteration:
        pass


def drop_until(pred_fn: Callable[[T], bool], iterable: Iterable[T]) -> Iterator[T]:
    r"""Drop elements from the iterable until an element that satisfies the predicate is encountered. Similar to the
    built-in :py:func:`filter` function, but only applied to a prefix of the iterable.

    .. code:: python

        >>> list(drop_until(lambda x: x > 5, range(10)))
        [6, 7, 8, 9]

    :param pred_fn: The predicate that returned elements should satisfy.
    :param iterable: The iterable.
    :return: The iterator after dropping elements.
    """
    iterator = iter(iterable)
    for item in iterator:
        if not pred_fn(item):
            continue
        yield item
        break
    yield from iterator


@overload
def split_by(iterable: Iterable[A], empty_segments: bool = False, *, criterion: Callable[[A], bool]) \
        -> Iterator[List[A]]: ...


@overload
def split_by(iterable: Iterable[A], empty_segments: bool = False, *, separator: A) \
        -> Iterator[List[A]]: ...


def split_by(iterable: Iterable[A], empty_segments: bool = False, *, criterion=None, separator=None) \
        -> Iterator[List[A]]:
    r"""Split a list into sub-lists by dropping certain elements. Exactly one of ``criterion`` and ``separator`` must be
    specified. For example:

    .. code:: python

        >>> list(split_by(range(10), criterion=lambda x: x % 3 == 0))
        [[1, 2], [4, 5], [7, 8]]

        >>> list(split_by(" Split by: ", empty_segments=True, separator='.'))
        [[], ['S', 'p', 'l', 'i', 't'], ['b', 'y', ':'], []]

    :param iterable: The list to split.
    :param empty_segments: If ``True``, include an empty list in cases where two adjacent elements satisfy
        the criterion.
    :param criterion: The criterion to decide whether to drop an element.
    :param separator: The separator for sub-lists. An element is dropped if it is equal to ``parameter``.
    :return: List of sub-lists.
    """
    if not ((criterion is None) ^ (separator is None)):
        raise ValueError("Exactly one of `criterion` and `separator` should be specified")
    if criterion is None:
        criterion = lambda x: x == separator
    group = []
    for x in iterable:
        if not criterion(x):
            group.append(x)
        else:
            if len(group) > 0 or empty_segments:
                yield group
            group = []
    if len(group) > 0 or empty_segments:
        yield group


@overload
def scanl(func: Callable[[A, A], A], iterable: Iterable[A]) -> Iterator[A]: ...


@overload
def scanl(func: Callable[[B, A], B], iterable: Iterable[A], initial: B) -> Iterator[B]: ...


def scanl(func, iterable, *args):
    r"""Computes the intermediate results of :py:func:`~functools.reduce`. Equivalent to Haskell's ``scanl``. For
    example:

    .. code:: python

        >>> list(scanl(operator.add, [1, 2, 3, 4], 0))
        [0, 1, 3, 6, 10]
        >>> list(scanl(lambda s, x: x + s, ['a', 'b', 'c', 'd']))
        ['a', 'ba', 'cba', 'dcba']

    Learn more at `Learn You a Haskell: Higher Order Functions <http://learnyouahaskell.com/higher-order-functions>`_.

    :param func: The function to apply. This should be a binary function where the arguments are: the accumulator,
        and the current element.
    :param iterable: The list of elements to iteratively apply the function to.
    :param initial: The initial value for the accumulator. If not supplied, the first element in the list is used.
    :return: The intermediate results at each step.
    """
    iterable = iter(iterable)
    if len(args) == 1:
        acc = args[0]
    elif len(args) == 0:
        acc = next(iterable)
    else:
        raise ValueError("Too many arguments")
    yield acc
    for x in iterable:
        acc = func(acc, x)
        yield acc


@overload
def scanr(func: Callable[[A, A], A], iterable: Iterable[A]) -> List[A]: ...


@overload
def scanr(func: Callable[[B, A], B], iterable: Iterable[A], initial: B) -> List[B]: ...


def scanr(func, iterable, *args):
    r"""Computes the intermediate results of :py:func:`~functools.reduce` applied in reverse. Equivalent to Haskell's
    ``scanr``. For example:

    .. code:: python

        >>> scanr(operator.add, [1, 2, 3, 4], 0)
        [10, 9, 7, 4, 0]
        >>> scanr(lambda s, x: x + s, ['a', 'b', 'c', 'd'])
        ['abcd', 'bcd', 'cd', 'd']

    Learn more at `Learn You a Haskell: Higher Order Functions <http://learnyouahaskell.com/higher-order-functions>`_.

    :param func: The function to apply. This should be a binary function where the arguments are: the accumulator,
        and the current element.
    :param iterable: The list of elements to iteratively apply the function to.
    :param initial: The initial value for the accumulator. If not supplied, the first element in the list is used.
    :return: The intermediate results at each step, starting from the end.
    """
    return list(scanl(func, reversed(iterable), *args))[::-1]


class LazyList(Generic[T], Sequence[T]):
    r"""A wrapper over an iterable to allow lazily converting it into a list. The iterable is only iterated up to the
    accessed indices.

    :param iterable: The iterable to wrap.
    """

    class LazyListIterator:
        def __init__(self, lst: 'LazyList[T]'):
            self.list = weakref.ref(lst)
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            try:
                obj = self.list()[self.index]
            except IndexError:
                raise StopIteration
            self.index += 1
            return obj

    def __init__(self, iterable: Iterable[T]):
        self.iter = iter(iterable)
        self.exhausted = False
        self.list: List[T] = []

    def __iter__(self):
        if self.exhausted:
            return iter(self.list)
        return self.LazyListIterator(self)

    def _fetch_until(self, idx: Optional[int]) -> None:
        if self.exhausted:
            return
        try:
            if idx is not None and idx < 0:
                idx = None  # otherwise we won't know when the sequence ends
            while idx is None or len(self.list) <= idx:
                self.list.append(next(self.iter))
        except StopIteration:
            self.exhausted = True
            del self.iter

    @overload
    def __getitem__(self, idx: int) -> T: ...

    @overload
    def __getitem__(self, idx: slice) -> List[T]: ...

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            self._fetch_until(idx.stop)
        else:
            self._fetch_until(idx)
        return self.list[idx]

    def __len__(self):
        if self.exhausted:
            return len(self.list)
        else:
            raise TypeError("__len__ is not available before the iterable is depleted")


class Range(Sequence[int]):
    r"""A replacement for built-in :py:class:`range` with support for indexing operators. For example:

    .. code:: python

        >>> r = Range(10)         # (end)
        >>> r = Range(1, 10 + 1)  # (start, end)
        >>> r = Range(1, 11, 2)   # (start, end, step)
        >>> print(r[0], r[2], r[4])
        1 5 9
    """

    @overload
    def __init__(self, stop: int): ...

    @overload
    def __init__(self, start: int, stop: int): ...

    @overload
    def __init__(self, start: int, stop: int, step: int): ...

    def __init__(self, *args):
        if len(args) == 0 or len(args) > 3:
            raise ValueError("Range should be called the same way as the builtin `range`")
        if len(args) == 1:
            self.l = 0
            self.r = args[0]
            self.step = 1
        else:
            self.l = args[0]
            self.r = args[1]
            self.step = 1 if len(args) == 2 else args[2]
        self.val = self.l
        self.length = (self.r - self.l) // self.step

    def __iter__(self) -> Iterator[int]:
        return Range(self.l, self.r, self.step)

    def __next__(self) -> int:
        if self.val >= self.r:
            raise StopIteration
        result = self.val
        self.val += self.step
        return result

    def __len__(self) -> int:
        return self.length

    def _get_idx(self, idx: int) -> int:
        return self.l + self.step * idx

    @overload
    def __getitem__(self, idx: int) -> int: ...

    @overload
    def __getitem__(self, idx: slice) -> List[int]: ...

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [self._get_idx(idx) for idx in range(*item.indices(self.length))]
        if item < 0:
            item = self.length + item
        return self._get_idx(item)


class MapList(Generic[R], Sequence[R]):
    r"""A wrapper over a list that allows lazily performing transformations on the list elements. It's basically the
    built-in :py:func:`map` function, with support for indexing operators. An example use case:

    .. code:: python

        >>> import bisect

        >>> # Find index of the first element in `a` whose square is >= 10.
        ... a = [1, 2, 3, 4, 5]
        ... pos = bisect.bisect_left(MapList(lambda x: x * x, a), 10)
        3

        >>> # Find the first index `i` such that `a[i] * b[i]` is >= 10.
        ... b = [2, 3, 4, 5, 6]
        ... pos = bisect.bisect_left(MapList(lambda i: a[i] * b[i], Range(len(a))), 10)
        2

    :param func: The transformation to perform on list elements.
    :param lst: The list to wrap.
    """

    def __init__(self, func: Callable[[T], R], lst: Sequence[T]):
        self.func = func
        self.list = lst

    @overload
    def __getitem__(self, idx: int) -> R: ...

    @overload
    def __getitem__(self, idx: slice) -> List[R]: ...

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.func(self.list[item])
        return [self.func(x) for x in self.list[item]]

    def __iter__(self) -> Iterator[R]:
        return map(self.func, self.list)

    def __len__(self) -> int:
        return len(self.list)
