import functools
import os
import pickle
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Iterator, Optional

from .log import log
from .types import PathType

__all__ = [
    "get_folder_size",
    "readable_size",
    "get_file_lines",
    "remove_prefix",
    "copy_tree",
    "cache",
    "scandir",
]

if platform.system() == "Darwin":
    def get_folder_size(path: PathType) -> int:
        # Credit: https://stackoverflow.com/a/25574638/4909228
        r"""Get disk usage of given path in bytes."""
        return int(subprocess.check_output(['du', '-s', str(path)],
                                           env={"BLOCKSIZE": "512"}).split()[0].decode('utf-8')) * 512
else:
    def get_folder_size(path: PathType) -> int:
        # Credit: https://stackoverflow.com/a/25574638/4909228
        r"""Get disk usage of given path in bytes."""
        return int(subprocess.check_output(['du', '-bs', str(path)]).split()[0].decode('utf-8'))


def readable_size(size: float) -> str:
    r"""Represent file size in human-readable format.

    .. code:: python

        >>> readable_size(1024 * 1024)
        "1.00M"

    :param size: File size in bytes.
    """
    units = ["", "K", "M", "G", "T", "P"]
    for unit in units[:-1]:
        if size < 1024:
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}{units[-1]}"


def get_file_lines(path: PathType) -> int:
    r"""Get number of lines in text file.
    """
    return int(subprocess.check_output(['wc', '-l', str(path)]).split()[0].decode('utf-8'))


def remove_prefix(s: str, prefix: str) -> str:
    r"""Remove the specified prefix from a string. If only parts of the prefix match, then only that part is removed.

    .. code:: python

        >>> remove_prefix("https://github.com/huzecong/flutes", "https://")
        "github.com/huzecong/flutes"

        >>> remove_prefix("preface", "prefix")
        "face"

    :param s: The string whose prefix we want to remove.
    :param prefix: The prefix to remove.
    """
    length = min(len(s), len(prefix))
    prefix_len = next((idx for idx in range(length) if s[idx] != prefix[idx]), length)
    return s[prefix_len:]


def copy_tree(src: PathType, dst: PathType, overwrite: bool = False) -> None:
    r"""Copy contents of folder ``src`` to folder ``dst``. The ``dst`` folder can exist or whatever (looking at you,
    :py:func:`shutil.copytree`).

    :param src: The source directory.
    :param dst: The destination directory. If it doesn't exist, it will be created.
    :param overwrite: If ``True``, files in ``dst`` will be overwritten if a file with the same relative path exists
        in ``src``. If ``False``, these files are not copied. Defaults to ``False``.
    """
    os.makedirs(dst, exist_ok=True)
    for file in os.listdir(src):
        src_path = os.path.join(src, file)
        dst_path = os.path.join(dst, file)
        if os.path.isdir(src_path):
            copy_tree(src_path, dst_path, overwrite=overwrite)
        elif overwrite or not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
    shutil.copystat(src, dst)


def cache(path: Optional[PathType], verbose: bool = True, name: Optional[str] = None):
    r"""A function decorator that caches the output of the function to disk. If the cache file exists, it is loaded from
    disk and the function will not be executed.

    :param path: Path to the cache file. If ``None``, no cache is loaded or saved.
    :param verbose: If ``True``, will print to log.
    :param name: Name of the object to load. Only used for logging purposes.
    """
    name = (name or 'cache').capitalize()

    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if path is not None and os.path.exists(path):
                with open(path, "rb") as f:
                    ret = pickle.load(f)
                if verbose:
                    log(f"{name} loaded from '{path}'")
            else:
                ret = func(*args, **kwargs)
                if path is not None:
                    with open(path, "wb") as f:
                        pickle.dump(ret, f)
                    if verbose:
                        log(f"{name} saved to '{path}'")
            return ret

        return wrapped

    return decorator


def scandir(path: PathType) -> Iterator[PathType]:
    r"""Lazily iterate over all files and directories under a directory. The returned path is the absolute path of the
    child file or directory, with the same type as :attr:`path` (:py:class:`pathlib.Path` or :py:class:`str`).

    :param path: Path to the directory.
    :return: An iterator over children paths.
    """
    if isinstance(path, Path):
        with os.scandir(path) as it:
            for entry in it:
                yield Path(entry.path)
    else:
        with os.scandir(path) as it:
            for entry in it:
                yield entry.path
