import operator

import flutes
from .utils import check_iterator


def test_chunk() -> None:
    check_iterator(flutes.chunk(range(10), 3),
                   [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]])
    check_iterator(flutes.chunk(range(5), 6),
                   [[0, 1, 2, 3, 4]])


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
    for i, x in enumerate(l[:50]):
        assert i == x
    assert len(l) == 100
    for i, x in enumerate(l):
        assert i == x
