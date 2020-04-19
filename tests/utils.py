from typing import Iterable, List, TypeVar

T = TypeVar('T')


def check_iterator(gen: Iterable[T], expected: List[T]) -> None:
    idx = 0
    for elem in gen:
        assert elem == expected[idx]
        idx += 1
    assert idx == len(expected)
