import functools
import multiprocessing as mp
from unittest.mock import NonCallableMagicMock, MagicMock

import flutes
from .utils import check_iterator


def sqr(x: int) -> int:
    return x * x


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


def progress_bar_fn(idx: int, bar) -> None:
    total = (idx + 1) * 2
    bar.new(desc=f"Bar {idx}", total=total)
    for i in range(total):
        bar.update(1, postfix={"i": i})
        if i % 5 == 1:
            flutes.log(f"test {i}")
    for i in bar.iter(range(total)):
        bar.update(postfix={"i": i})


def test_ProgressBarManager() -> None:
    for proc in [0, 2]:
        # Test multiprocessing in `proc = 2`
        # Test coverage in `proc = 0`
        manager = flutes.ProgressBarManager()
        with flutes.safe_pool(proc, closing=[manager]) as pool:
            fn = functools.partial(progress_bar_fn, bar=manager.proxy)
            pool.map(fn, range(10))
