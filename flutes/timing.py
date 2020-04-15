import contextlib
import time

__all__ = [
    "work_in_progress",
]


@contextlib.contextmanager
def work_in_progress(msg: str):
    print(msg + "... ", end='', flush=True)
    begin_time = time.time()
    yield
    time_consumed = time.time() - begin_time
    print(f"done. ({time_consumed:.2f}s)")
