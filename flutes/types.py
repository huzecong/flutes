from pathlib import Path
from typing import Dict, List, Sequence, Tuple, TypeVar, Union

__all__ = [
    'MaybeTuple',
    'MaybeList',
    'MaybeSeq',
    'MaybeDict',
    'PathType',
]

T = TypeVar('T')
MaybeTuple = Union[T, Tuple[T, ...]]
MaybeList = Union[T, List[T]]
MaybeSeq = Union[T, Sequence[T]]
MaybeDict = Union[T, Dict[str, T]]
PathType = TypeVar('PathType', str, Path)
