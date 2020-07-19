from pathlib import Path
from typing import Callable, Dict, List, Sequence, TYPE_CHECKING, Tuple, TypeVar, Union

if TYPE_CHECKING:
    from tqdm import tqdm

__all__ = [
    "MaybeTuple",
    "MaybeList",
    "MaybeSeq",
    "MaybeDict",
    "PathType",
    "BarFn",
]

T = TypeVar('T')
MaybeTuple = Union[T, Tuple[T, ...]]
MaybeList = Union[T, List[T]]
MaybeSeq = Union[T, Sequence[T]]
MaybeDict = Union[T, Dict[str, T]]
PathType = TypeVar('PathType', str, Path)
BarFn = Callable[..., 'tqdm']
