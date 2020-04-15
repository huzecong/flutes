from typing import Iterator, TypeVar, List

T = TypeVar('T')


def check_iterator(gen: Iterator[T], expected: List[T]) -> None:
    idx = 0
    for elem in gen:
        assert elem == expected[idx]
        idx += 1
    assert idx == len(expected)
