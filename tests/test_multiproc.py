import functools
import itertools
import multiprocessing as mp
import os
import tempfile
import time
from typing import Dict, List, Tuple, Iterator
from unittest.mock import MagicMock, NonCallableMagicMock, patch

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
        check_iterator(pool.imap(sqr, seq), target)
    assert not isinstance(pool, pool_type)
    file_obj.assert_called_once()

    file_obj = NonCallableMagicMock()
    file_obj.mock_add_spec(["close"])
    with flutes.safe_pool(2, closing=[file_obj], suppress_exceptions=True) as pool:
        result = list(pool.imap(sqr, seq))
        raise ValueError  # should swallow exceptions
    assert isinstance(pool, pool_type)
    assert result == target
    file_obj.close.assert_called_once()

    with pytest.raises(KeyboardInterrupt):
        with flutes.safe_pool(2, closing=[file_obj], suppress_exceptions=True) as pool:
            raise KeyboardInterrupt


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

    def generate(self, start: int, stop: int, *args) -> None:
        for x in range(start, stop):
            self.numbers.append(x)

    def gather_fn(self, bounds: Tuple[int, int]) -> Iterator[int]:
        l, r = bounds
        yield from range(l, r)


def test_stateful_pool_get_state() -> None:
    for n_procs in [0, 2]:
        with flutes.safe_pool(n_procs, state_class=PoolState2) as pool:
            intervals = list(range(0, 100 + 1, 5))
            pool.starmap(PoolState2.generate, zip(intervals, intervals[1:]), args=(1, 2))  # dummy args
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
                assert pool.map(sqr, seq, args=args, kwds=kwds) == target
                check_iterator(pool.imap(sqr, seq, args=args, kwds=kwds), target)
                assert sorted(pool.imap_unordered(sqr, seq, args=args, kwds=kwds)) == target
                assert pool.starmap(mul, zip(seq, seq), args=args, kwds=kwds) == target
                assert pool.map_async(sqr, seq, args=args, kwds=kwds).get() == target
                assert pool.starmap_async(mul, zip(seq, seq), args=args, kwds=kwds).get() == target
                assert pool.apply(sqr, (10, 2), kwds=kwds) == 100 * 2 * 3
                assert pool.apply_async(sqr, (10, 2), kwds=kwds).get() == 100 * 2 * 3


def gather_fn(bounds: Tuple[int, int]) -> Iterator[int]:
    l, r = bounds
    yield from range(l, r)


def test_gather() -> None:
    n = 10000
    intervals = list(range(0, n + 1, 1000))
    answer = set(range(n))
    for n_procs in [0, 2]:
        with flutes.safe_pool(n_procs) as pool:
            assert set(pool.gather(gather_fn, zip(intervals, intervals[1:]))) == answer
        with flutes.safe_pool(n_procs, state_class=PoolState2) as pool:
            assert set(pool.gather(PoolState2.gather_fn, zip(intervals, intervals[1:]))) == answer


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
    for verbose in [False, True]:
        for proc in [0, 2]:
            # Test multiprocessing in `proc = 2`
            # Test coverage in `proc = 0`
            manager = flutes.ProgressBarManager(verbose=verbose)
            with flutes.safe_pool(proc, closing=[manager]) as pool:
                fn = functools.partial(progress_bar_fn, bar=manager.proxy)
                pool.map(fn, range(10))
                fn = functools.partial(file_progress_bar_fn, bar=manager.proxy)
                pool.map(fn, range(4))
            flutes.log(f"This should still show up: verbose={verbose}, proc={proc}", force_console=True)


def test_ProgressBarManager_increment_correct() -> None:
    call_count = progress = 0

    def update(value: int):
        nonlocal call_count, progress
        call_count += 1
        progress += value

    manager = flutes.ProgressBarManager()
    proxy = manager.proxy
    proxy.update = update

    for _ in proxy.new(list(range(10000)), update_frequency=0.02):
        pass
    assert call_count == int(1.0 / 0.02) and progress == 10000

    call_count = progress = 0
    for _ in proxy.new(list(range(5)), update_frequency=0.01):
        pass
    assert call_count == 5 and progress == 5

    call_count = progress = 0
    for _ in proxy.new(range(10020), update_frequency=50):
        pass
    assert call_count == flutes.ceil_div(10020, 50) and progress == 10020

    call_count = progress = 0
    for _ in proxy.new(range(2), update_frequency=50):
        pass
    assert call_count == 1 and progress == 2
