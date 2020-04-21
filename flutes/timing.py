import contextlib
import time

__all__ = [
    "work_in_progress",
]


@contextlib.contextmanager
def work_in_progress(desc: str = "Work in progress"):
    r"""Time the execution time of a code block or function.

    .. code:: python

        >>> @work_in_progress("Loading file")
        ... def load_file(path):
        ...     with open(path, "rb") as f:
        ...         return pickle.load(f)
        ...
        ... obj = load_file("/path/to/some/file")
        Loading file... done. (3.52s)

        >>> with work_in_progress("Saving file"):
        ...     with open(path, "wb") as f:
        ...         pickle.dump(obj, f)
        Saving file... done. (3.78s)

    :param desc: Description of the task performed.
    """
    print(desc + "... ", end='', flush=True)
    begin_time = time.time()
    yield
    time_consumed = time.time() - begin_time
    print(f"done. ({time_consumed:.2f}s)")
