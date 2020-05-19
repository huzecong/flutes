import functools
import itertools
import multiprocessing as mp
import os
import tempfile
import time
from typing import Dict, List
from unittest.mock import MagicMock, NonCallableMagicMock

import pytest

import flutes
from .utils import check_iterator


def sqr(x: int, coef: int = 1, *, coef2: int = 1) -> int:
    return x * x * coef * coef2


def mul(x: int, y: int, coef: int = 1, *, coef2: int = 1) -> int:
    return x * y * coef * coef2


def test_safe_pool() -> None:
    seq = list(range(10000))
    target = list(map(sqr, seq))  # sequential
    with mp.Pool(1) as pool:
        pool_type = type(pool)

    file_obj = MagicMock()
    with flutes.safe_pool(0, closing=[file_obj]) as pool:
        assert not isinstance(pool, pool_type)
        check_iterator(pool.imap(sqr, seq), target)
    file_obj.assert_called_once()

    file_obj = NonCallableMagicMock()
    file_obj.mock_add_spec(["close"])
    with flutes.safe_pool(2, closing=[file_obj]) as pool:
        assert isinstance(pool, pool_type)
        check_iterator(pool.imap(sqr, seq), target)
        raise ValueError  # should swallow exceptions
    file_obj.close.assert_called_once()


class PoolState(flutes.PoolState):
    CLASS_VAR: int = 1

    def __init__(self, large_dict: Dict[str, int]):
        self.large_dict = large_dict

    def _some_func(self, x: str) -> int:
        return self.large_dict[x]

    def convert(self, x: str) -> int:
        return self._some_func(x) + self.CLASS_VAR


def test_stateful_pool() -> None:
    large_dict = {str(i): i for i in range(100000)}
    seq = list(map(str, range(100000)))
    target = sum(map(lambda x: int(x) + 1, seq))  # sequential

    for n_procs in [0, 2]:
        with flutes.safe_pool(n_procs, state_class=PoolState, init_args=(large_dict,)) as pool:
            result = sum(pool.imap_unordered(PoolState.convert, seq, chunksize=1000))

            # See, if you had a type checker, you wouldn't be making these mistakes.
            with pytest.raises(ValueError, match="Bound methods of the pool state class"):
                _ = sum(pool.imap_unordered(PoolState({}).convert, seq, chunksize=1000))  # type: ignore[arg-type]
            with pytest.raises(ValueError, match="Only unbound methods of the pool state class"):
                _ = sum(pool.imap_unordered(PoolState2.generate, seq, chunksize=1000))  # type: ignore[arg-type]
        assert result == target


class PoolState2(flutes.PoolState):
    def __init__(self):
        self.numbers: List[int] = []

    def __return_state__(self):
        return self.numbers

    def generate(self, start: int, stop: int) -> None:
        for x in range(start, stop):
            self.numbers.append(x)


def test_stateful_pool_get_state() -> None:
    for n_procs in [0, 2]:
        with flutes.safe_pool(n_procs, state_class=PoolState2) as pool:
            intervals = list(range(0, 100 + 1, 5))
            pool.starmap(PoolState2.generate, zip(intervals, intervals[1:]))
            states = pool.get_states()
            assert sorted(itertools.chain.from_iterable(states)) == list(range(100))  # type: ignore[arg-type]


def test_pool_methods() -> None:
    seq = list(range(10000))
    args = (2,)
    kwds = {"coef2": 3}
    target = [sqr(x, *args, **kwds) for x in seq]  # sequential
    for n_procs in [0, 2]:
        for state_class in [PoolState, None]:
            with flutes.safe_pool(n_procs, state_class=state_class, init_args=(None,)) as pool:
                map_result = pool.map(sqr, seq, args=args, kwds=kwds)
                imap_result = pool.imap(sqr, seq, args=args, kwds=kwds)
                imap_unordered_result = sorted(pool.imap_unordered(sqr, seq, args=args, kwds=kwds))
                starmap_result = pool.starmap(mul, zip(seq, seq), args=args, kwds=kwds)
                map_async_result = pool.map_async(sqr, seq, args=args, kwds=kwds).get()
                starmap_async_result = pool.starmap_async(mul, zip(seq, seq), args=args, kwds=kwds).get()
                apply_result = pool.apply(sqr, (10, 2), kwds=kwds)
                apply_async_result = pool.apply_async(sqr, (10, 2), kwds=kwds).get()
            for result in [map_result, imap_result, imap_unordered_result, starmap_result, map_async_result,
                           starmap_async_result]:
                check_iterator(result, target)
            for result in [apply_result, apply_async_result]:
                assert result == 100 * 2 * 3


def progress_bar_fn(idx: int, bar) -> None:
    total = (idx + 1) * 2
    bar.new(desc=f"Bar {idx}", total=total)
    for i in range(total):
        bar.update(1, postfix={"i": i})
        if i % 5 == 1:
            flutes.log(f"test {i}")
    for i in bar.new(range(total)):
        bar.update(postfix={"i": i})


def file_progress_bar_fn(idx: int, bar) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "file.txt")
        with open(path, "w") as f:
            f.write("\n".join(map(str, range(100))))
        with flutes.progress_open(path, bar_fn=bar.new, desc=f"Bar {idx}") as fin:
            bar.update(postfix={"path": path})
            for line in fin:
                time.sleep(0.01)


def test_ProgressBarManager() -> None:
    for proc in [0, 2]:
        # Test multiprocessing in `proc = 2`
        # Test coverage in `proc = 0`
        manager = flutes.ProgressBarManager()
        with flutes.safe_pool(proc, closing=[manager]) as pool:
            fn = functools.partial(progress_bar_fn, bar=manager.proxy)
            pool.map(fn, range(10))
            fn = functools.partial(file_progress_bar_fn, bar=manager.proxy)
            pool.map(fn, range(4))
    flutes.log("This should still show up", force_console=True)
