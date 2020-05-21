import operator

import pytest

import flutes
from .utils import check_iterator


def test_chunk() -> None:
    check_iterator(flutes.chunk(3, range(10)),
                   [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]])
    check_iterator(flutes.chunk(6, range(5)),
                   [[0, 1, 2, 3, 4]])


def test_take() -> None:
    check_iterator(flutes.take(5, range(10000000)),
                   [0, 1, 2, 3, 4])
    check_iterator(flutes.take(5, range(2)),
                   [0, 1])


def test_drop() -> None:
    check_iterator(flutes.drop(5, range(10)),
                   [5, 6, 7, 8, 9])
    check_iterator(flutes.drop(5, range(2)),  # type: ignore[misc]
                   [])


def test_drop_until() -> None:
    check_iterator(flutes.drop_until(lambda x: x > 5, range(10)),
                   [6, 7, 8, 9])


def test_split_by() -> None:
    check_iterator(flutes.split_by(range(10), criterion=lambda x: x % 3 == 0),
                   [[1, 2], [4, 5], [7, 8]])
    check_iterator(flutes.split_by(" Split by: ", empty_segments=True, separator=' '),
                   [[], ['S', 'p', 'l', 'i', 't'], ['b', 'y', ':'], []])


def test_scanl() -> None:
    check_iterator(flutes.scanl(operator.add, [1, 2, 3, 4], 0),
                   [0, 1, 3, 6, 10])
    check_iterator(flutes.scanl(lambda s, x: x + s, ['a', 'b', 'c', 'd']),
                   ['a', 'ba', 'cba', 'dcba'])


def test_scanr() -> None:
    check_iterator(flutes.scanr(operator.add, [1, 2, 3, 4], 0),
                   [10, 9, 7, 4, 0])
    check_iterator(flutes.scanr(lambda s, x: x + s, ['a', 'b', 'c', 'd']),
                   ['abcd', 'bcd', 'cd', 'd'])


def test_LazyList() -> None:
    l = flutes.LazyList(range(100))
    assert l[50] == 50
    assert l[70:90] == list(range(70, 90))
    assert l[-2] == 98

    l = flutes.LazyList(range(100))
    with pytest.raises(TypeError, match="__len__"):
        len(l)
    for i, x in enumerate(l):
        assert i == x
    assert len(l) == 100
    for i, x in enumerate(l):
        assert i == x


def test_Range() -> None:
    def _check_range(*args):
        r = flutes.Range(*args)
        gold = list(range(*args))
        assert len(r) == len(gold)
        check_iterator(r, gold)
        assert r[1:-1] == gold[1:-1]
        assert r[-2] == gold[-2]

    _check_range(10)
    _check_range(1, 10 + 1)
    _check_range(1, 11, 2)


def test_MapList() -> None:
    l = flutes.MapList(lambda x: x * x, list(range(100)))
    assert l[15] == 15 * 15
    check_iterator(l[20:-10], [x * x for x in range(20, 100 - 10)])
    assert len(l) == 100
    check_iterator(l, [x * x for x in range(100)])

    import bisect
    a = [1, 2, 3, 4, 5]
    pos = bisect.bisect_left(flutes.MapList(lambda x: x * x, a), 10)
    assert pos == 3
    b = [2, 3, 4, 5, 6]
    pos = bisect.bisect_left(flutes.MapList(lambda i: a[i] * b[i], flutes.Range(len(a))), 10)
    assert pos == 2
