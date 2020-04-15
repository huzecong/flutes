import weakref
from typing import Callable, Generic, Iterable, Iterator, List, Optional, TypeVar, overload

__all__ = [
    "chunk",
    "drop_until",
    "split_by",
    "scanl",
    "scanr",
    "LazyList",
]

T = TypeVar('T')
A = TypeVar('A')
B = TypeVar('B')
R = TypeVar('R')


def chunk(iterable: Iterable[T], n: int) -> Iterator[List[T]]:
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


def drop_until(pred_fn: Callable[[T], bool], iterable: Iterable[T]) -> Iterator[T]:
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
    specified. For example::

    .. code-block:: python

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


# This is what happens when you don't have the Haskell `scanl`
@overload
def scanl(func: Callable[[A, A], A], iterable: Iterable[A]) -> Iterator[A]: ...


@overload
def scanl(func: Callable[[B, A], B], iterable: Iterable[A], initial: B) -> Iterator[B]: ...


def scanl(func, iterable, *args):
    r"""Computes the intermediate results of :meth:`reduce`. Equivalent to Haskell's ``scanl``. For example:

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
    r"""Computes the intermediate results of :meth:`reduce` applied in reverse. Equivalent to Haskell's ``scanr``.
    For example:

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


class LazyList(Generic[T]):
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

    def __init__(self, iterator: Iterable[T]):
        self.iter = iter(iterator)
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
            while idx is None or len(self.list) <= idx:
                self.list.append(next(self.iter))
        except StopIteration:
            self.exhausted = True
            del self.iter

    @overload
    def __getitem__(self, idx: int) -> T:
        ...

    @overload
    def __getitem__(self, idx: slice) -> List[T]:
        ...

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            self._fetch_until(idx.stop)
        else:
            self._fetch_until(idx)
        return self.list[idx]

    def __len__(self):
        self._fetch_until(None)
        return len(self.list)
