import random
from collections import OrderedDict
from typing import NamedTuple

import flutes


def test_reverse_map() -> None:
    seq = list(range(100))
    random.shuffle(seq)
    d = {x: i for i, x in enumerate(seq)}
    assert flutes.reverse_map(d) == seq


def test_map_structure() -> None:
    class Example(NamedTuple):
        a: int
        b: float
        c: str

    obj = {
        "a": [1, 2, OrderedDict([("b", "c"), (1, 2)])],
        (1, 2): {"a", "b", 4},
        "3": Example(a=1, b=2.2, c="c"),
        4: (5, 6, 7),
    }

    target = {
        "a": [2, 4, OrderedDict([("b", "cc"), (1, 4)])],
        (1, 2): {"aa", "bb", 8},
        "3": Example(a=2, b=4.4, c="cc"),
        4: (10, 12, 14),
    }
    output = flutes.map_structure(lambda x: x * 2, obj)
    assert output == target

    def fn(x):
        if isinstance(x, int):
            l = flutes.no_map_instance(tuple(range(x)))
            return l
        return x

    del obj[(1, 2)]  # remove the set
    del target[(1, 2)]
    obj2 = flutes.map_structure(fn, obj)

    def fn2(b, a):
        if isinstance(b, tuple):
            return a + len(b)
        return a + b

    output = flutes.map_structure_zip(fn2, (obj2, obj))
    assert output == target
