import contextlib
import functools
import inspect
import multiprocessing as mp
import pickle
import sys
import threading
import time
import traceback
import types
from collections import defaultdict
from multiprocessing.pool import Pool
from queue import Empty
from types import FrameType
from typing import (Any, Callable, Dict, Generic, IO, Iterable, Iterator, List, Mapping, NamedTuple, Optional, Set,
                    Tuple, Type, TypeVar, Union, cast, no_type_check, overload)

from multiprocessing.reduction import AbstractReducer
from tqdm import tqdm
from typing_extensions import Literal

from .exception import log_exception
from .log import get_worker_id
from .types import PathType

__all__ = [
    "safe_pool",
    "PoolState",
    "MultiprocessingFileWriter",
    "kill_proc_tree",
    "ProgressBarManager",
]

T = TypeVar('T')
R = TypeVar('R')


class DummyApplyResult(mp.pool.ApplyResult, Generic[T]):
    def __init__(self, value: T):
        self._value = value

    def ready(self) -> bool:
        return True

    def success(self) -> bool:
        return True

    def wait(self, timeout: Optional[float] = None) -> None:
        pass

    def get(self, timeout: Optional[float] = None) -> T:
        return self._value


class DummyPool:
    r"""A wrapper over ``multiprocessing.Pool`` that uses single-threaded execution when :attr:`processes` is zero.
    """
    _state: int

    def __init__(self, processes: Optional[int] = None, initializer: Optional[Callable[..., None]] = None,
                 initargs: Iterable[Any] = (), maxtasksperchild: Optional[int] = None,
                 context: Optional[Any] = None) -> None:
        self._process_state = None
        if initializer is not None:
            # A hack to accomodate stateful pools.
            def run_initializer():
                initializer(*initargs)
                return locals()

            self._process_state = run_initializer().get("__state__", None)

        self._state = mp.pool.RUN

    def imap(self, fn: Callable[[T], R], iterable: Iterable[T], *_, args=(), kwds={}, **__) -> Iterator[R]:
        if self._process_state is not None:
            locals().update({"__state__": self._process_state})
        for x in iterable:
            yield fn(x, *args, **kwds)  # type: ignore[call-arg]

    def imap_unordered(self, fn: Callable[[T], R], iterable: Iterable[T], *_, args=(), kwds={}, **__) -> Iterator[R]:
        return self.imap(fn, iterable, args=args, kwds=kwds)

    def map(self, fn: Callable[[T], R], iterable: Iterable[T], *_, args=(), kwds={}, **__) -> List[R]:
        return list(self.imap(fn, iterable, args=args, kwds=kwds))

    def map_async(self, fn: Callable[[T], R], iterable: Iterable[T], *_, args=(), kwds={}, **__) \
            -> 'mp.pool.ApplyResult[List[R]]':
        return DummyApplyResult(self.map(fn, iterable, args=args, kwds=kwds))

    def starmap(self, fn: Callable[..., R], iterable: Iterable[Tuple[T, ...]], *_, args=(), kwds={}, **__) -> List[R]:
        if self._process_state is not None:
            locals().update({"__state__": self._process_state})
        return [fn(*x, *args, **kwds) for x in iterable]

    def starmap_async(self, fn: Callable[..., R], iterable: Iterable[Tuple[T, ...]], *_, args=(), kwds={}, **__) \
            -> 'mp.pool.ApplyResult[List[R]]':
        return DummyApplyResult(self.starmap(fn, iterable, args=args, kwds=kwds))

    def apply(self, fn: Callable[..., R], args: Iterable[Any] = (), kwds: Dict[str, Any] = {}, *_, **__) -> R:
        if self._process_state is not None:
            locals().update({"__state__": self._process_state})
        return fn(*args, **kwds)

    def apply_async(self, fn: Callable[..., R], args: Iterable[Any] = (), kwds: Dict[str, Any] = {}, *_, **__) \
            -> 'mp.pool.ApplyResult[R]':
        return DummyApplyResult(self.apply(fn, args, kwds))

    def gather(self, fn: Callable[[T], Iterator[R]], iterable: Iterable[T], *_, args=(), kwds={}, **__) -> Iterator[R]:
        if self._process_state is not None:
            locals().update({"__state__": self._process_state})
        for x in iterable:
            yield from fn(x, *args, **kwds)  # type: ignore[call-arg]

    @staticmethod
    def _no_op(self, *args, **kwargs):
        pass

    def __getattr__(self, item):
        return types.MethodType(DummyPool._no_op, self)  # no-op for everything else

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._state = mp.pool.TERMINATE


class PoolState:
    r"""Base class for multi-processing pool states. Pool states are mutable objects stored on each worker process, it
    allows keeping track of an process-local internal state that persists through tasks. This extends the capabilities
    of pool tasks beyond pure functions --- side-effects can also be recorded.

    To define a pool state, subclass the :class:`PoolState` class and define the ``__init__`` method, which will be
    called when each worker process is spawn. Methods of the state class can then be used as pool tasks.

    Here's an comprehensive example that reads a text file and counts the frequencies for each word appearing in the
    file. We use a map-reduce approach that distributes tasks to each pool worker and then "reduces" (aggregates) the
    results.

    .. code:: python

        class WordCounter(flutes.PoolState):
            def __init__(self):
                # Initializes the state; will be called when a worker process is spawn.
                self.word_cnt = collections.Counter()

            @flutes.exception_wrapper()  # prevent the worker from crashing, thus losing data
            def count_words(self, sentence):
                self.word_cnt.update(sentence.split())

        def count_words_in_file(path):
            with open(path) as f:
                lines = [line for line in f]
            # Construct a process pool while specifying the pool state class we're using.
            with flutes.safe_pool(processes=4, state_class=WordCounter) as pool:
                # Map the tasks as usual.
                for _ in pool.imap_unordered(WordCounter.count_words, sentences, chunksize=1000):
                    pass
                word_counter = collections.Counter()
                # Gather the states and perform the reduce step.
                for state in pool.get_states():
                    word_counter.update(state.word_cnt)
            return word_counter

    **See also:** :func:`safe_pool`, :class:`StatefulPoolType`.
    """
    __broadcasted__: bool

    def __return_state__(self):
        r"""When :meth:`StatefulPoolType.get_states` is invoked, this method is called for each pool worker to return
        its state. The default implementation returns the :class:`PoolState` object itself, but it might be beneficial
        to override this method in cases such as:

        - The :class:`PoolState` object contains unpickle-able attributes.
        - You need to dynamically compute the state before it's retrieved.
        """
        return self


def _pool_state_init(state_class: Type[PoolState], *args, **kwargs) -> None:
    # Wrapper for initializer function passed to stateful pools.
    state_obj = state_class(*args, **kwargs)  # type: ignore[call-arg]
    # _pool_state_init -> worker
    local_vars = inspect.currentframe().f_back.f_locals  # type: ignore[union-attr]
    local_vars['__state__'] = state_obj
    del local_vars


def _pool_fn_with_state(fn: Callable[..., R], *args, **kwds) -> R:
    # Wrapper for compute function passed to stateful pools.
    frame = cast(FrameType, inspect.currentframe().f_back)  # type: ignore[union-attr]
    while '__state__' not in frame.f_locals:  # the function might be wrapped several types
        frame = cast(FrameType, frame.f_back)  # _pool_fn_with_state -> mapper -> worker
    local_vars = frame.f_locals
    state_obj = local_vars['__state__']
    del frame, local_vars
    return fn(state_obj, *args, **kwds)


def _chain_fns(fns: List[Callable[..., R]], fn_arg_kwargs: List[Tuple[Tuple[Any, ...], Dict[str, Any]]]) -> List[R]:
    rets = []
    for fn, (args, kwargs) in zip(fns, fn_arg_kwargs):
        rets.append(fn(*args, **kwargs))
    return rets


State = TypeVar('State', bound=PoolState)


class FuncWrapper:
    def __init__(self, fn: Callable[..., R], args: Iterable[Any], kwds: Mapping[str, Any]):
        self.fn = fn
        self.args = args
        self.kwds = kwds

    def __call__(self, *args):
        return self.fn(*args, *self.args, **self.kwds)


# END_SIGNATURE = (random.random(), b"END")  # this would break if start_method is "spawn"
END_SIGNATURE = (b"END",)


def _gather_fn(queue: 'mp.Queue[R]', fn: Callable[[T], Iterator[R]], *args, **kwargs) -> Optional[bool]:
    try:
        for x in fn(*args, **kwargs):  # type: ignore[call-arg]
            queue.put(x)
    except Exception as e:
        log_exception(e)
    # No matter what happens, signal the end of generation.
    queue.put(cast(R, END_SIGNATURE))
    return True


class CustomMPReducer(AbstractReducer):
    class ForkingPickler(AbstractReducer.ForkingPickler):
        def __init__(self, *args, **kwargs):
            # Override argument to always use the highest protocol.
            if len(args) >= 2:
                args = (args[0], pickle.HIGHEST_PROTOCOL, *args[2:])
            else:
                kwargs["protocol"] = pickle.HIGHEST_PROTOCOL
            super().__init__(*args, **kwargs)


class PoolWrapper(mp.pool.Pool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Patch every method except `apply` and `apply_async`.
        for name in ["imap", "imap_unordered", "map", "map_async", "starmap", "starmap_async"]:
            pool_method = getattr(self, name)
            wrapped_method = self._define_method(pool_method)
            setattr(self, name, wrapped_method)

    @staticmethod
    def _define_method(pool_method: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(pool_method)
        def wrapped_method(func, *_, args=(), kwds={}, **__):
            if len(args) > 0 or len(kwds) > 0:
                func = FuncWrapper(func, args, kwds)
            return pool_method(func, *_, **__)

        return wrapped_method

    def gather(self, fn: Callable[[T], Iterator[R]], iterable: Iterable[T], chunksize: int = 1,
               args: Iterable[Any] = (), kwds: Dict[str, Any] = {}) -> Iterator[R]:
        # Refer to documentation at `PoolType.gather`.
        ctx = mp.get_context()
        ctx.reducer = CustomMPReducer  # type: ignore[assignment]
        with ctx.Manager() as manager:
            queue = manager.Queue()
            gather_fn = functools.partial(_gather_fn, queue, fn)
            if not isinstance(iterable, list):
                iterable = list(iterable)
            length = len(iterable)
            end_count = 0
            ret = self.map_async(  # type: ignore[call-arg]
                gather_fn, iterable, chunksize=chunksize, args=args, kwds=kwds)
            while True:
                try:
                    x = queue.get_nowait()
                except Empty:
                    if ret.ready():
                        # Update length to the number of end signatures successfully returned.
                        new_length = sum(map(bool, ret.get()))
                        if end_count == new_length:
                            break
                        length = new_length
                    time.sleep(0.1)  # queue empty, wait for a bit
                    continue
                except (OSError, ValueError):
                    break  # data in queue could be corrupt, e.g. if worker process is terminated while enqueueing
                if x == END_SIGNATURE:
                    end_count += 1
                    if end_count == length:
                        break
                else:
                    yield x


class StatefulPool(Generic[State]):
    _pool: 'PoolType'
    _state_class: Type[State]
    _class_methods: Set[int]  # a list of addresses of instance methods for the state class

    def __init__(self, pool_class: Type['PoolType'], state_class: Type[State], state_init_args: Tuple[Any, ...],
                 args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        self._state_class = state_class

        # Store the IDs of all methods of the `PoolState` subclass.
        self._class_methods = set()
        for attr_name in dir(self._state_class):
            attr_val = getattr(self._state_class, attr_name)
            if inspect.isfunction(attr_val):
                self._class_methods.add(id(attr_val))

        def get_arg(pos: int, name: str, default=None):
            if len(args) > pos + 1:
                return args[pos]
            if name in kwargs:
                return kwargs[name]
            return default

        def set_arg(pos: int, name: str, val):
            nonlocal args
            if len(args) > pos + 1:
                args = args[:pos] + (val,) + args[(pos + 1):]
            else:
                kwargs[name] = val

        state_init_fn = functools.partial(_pool_state_init, state_class)
        # If there's a user-defined initializer function...
        initializer = get_arg(1, "initializer", None)
        init_args = get_arg(2, "initargs", ())
        if initializer is not None:
            initializer = functools.partial(_chain_fns, fns=[state_init_fn, initializer])
            init_args = [(state_init_args, {}), (init_args, {})]
        else:
            initializer = state_init_fn
            init_args = state_init_args
        set_arg(1, "initializer", initializer)
        set_arg(2, "initargs", init_args)

        self._pool = pool_class(*args, **kwargs)

        for name in ["imap", "imap_unordered", "map", "map_async", "starmap", "starmap_async",
                     "apply", "apply_async", "gather"]:
            pool_method = getattr(self._pool, name)
            wrapped_method = self._define_method(pool_method)
            setattr(self, name, wrapped_method)

    @no_type_check
    def _wrap_fn(self, func: Callable[[State, T], R], allow_function: bool = True) -> Callable[[T], R]:
        # If the function is a `PoolState` method, wrap it to allow access to `self`.
        if id(func) in self._class_methods:
            return functools.partial(_pool_fn_with_state, func)
        if inspect.ismethod(func):
            if func.__self__.__class__ is self._state_class:
                raise ValueError(f"Bound methods of the pool state class {self._state_class.__name__} are not "
                                 f"accepted; use an unbound method instead.")
        if not allow_function:
            raise ValueError(f"Only unbound methods of the pool state class {self._state_class.__name__} are accepted")
        if inspect.isfunction(func):
            args = inspect.getfullargspec(func)
            if len(args.args) > 0 and args.args[0] == "self":
                raise ValueError(f"Only unbound methods of the pool state class {self._state_class.__name__} are "
                                 f"accepted")
        return func

    def _define_method(self, pool_method: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(pool_method)
        def wrapped_method(func, *args, **kwargs):
            return pool_method(self._wrap_fn(func), *args, **kwargs)

        return wrapped_method

    @staticmethod
    def _init_broadcast(self: State, _dummy: int) -> int:
        self.__broadcasted__ = False
        worker_id = get_worker_id()
        assert worker_id is not None
        return worker_id

    @staticmethod
    def _apply_broadcast(self: State, broadcast_fn: Callable[[State], R], *args, **kwds) -> Optional[Tuple[R, int]]:
        if not hasattr(self, '__broadcasted__'):
            # Might be possible that a worker crashed and restarted.
            self.__broadcasted__ = False
        if self.__broadcasted__:
            return None
        self.__broadcasted__ = True
        worker_id = get_worker_id()
        assert worker_id is not None
        result = broadcast_fn(self, *args, **kwds)  # type: ignore[call-arg]
        return (result, worker_id)

    def get_states(self) -> List[State]:
        r"""Return the states of each pool worker.

        :return: A list of state for each worker process. Order is arbitrary.
        """
        return self.broadcast(self._state_class.__return_state__)

    def broadcast(self, fn: Callable[[State], R], *, args: Iterable[Any] = (), kwds: Mapping[str, Any] = {}) -> List[R]:
        r"""Broadcast a function to each pool worker, and gather results.

        :param fn: The function to broadcast.
        :param args: Positional arguments to apply to the function.
        :param kwds: Keyword arguments to apply to the function.
        :return: The broadcast result from each worker process. Order is arbitrary.
        """
        if self._pool._state != mp.pool.RUN:
            raise ValueError("Pool not running")
        _ = self._wrap_fn(fn, allow_function=False)  # ensure that the function is an unbound method
        if isinstance(self._pool, DummyPool):
            return [fn(self._pool._process_state, *args, **kwds)]
        assert isinstance(self._pool, Pool)

        # Initialize the worker states.
        received_ids: Set[int] = set()
        n_processes = self._pool._processes
        broadcast_init_fn = functools.partial(_pool_fn_with_state, self._init_broadcast)
        while len(received_ids) < n_processes:
            init_ids: List[int] = self._pool.map(broadcast_init_fn, range(n_processes))  # type: ignore[arg-type]
            received_ids.update(init_ids)

        # Perform broadcast.
        received_ids: Set[int] = set()
        broadcast_results = []
        broadcast_handler_fn = functools.partial(_pool_fn_with_state, self._apply_broadcast)
        while len(received_ids) < n_processes:
            results: List[Optional[Tuple[R, int]]] = self._pool.map(
                broadcast_handler_fn, [fn] * n_processes, args=args, kwds=kwds)  # type: ignore[arg-type]
            for result in results:
                if result is not None:
                    ret, worker_id = result
                    received_ids.add(worker_id)
                    broadcast_results.append(ret)
        return broadcast_results

    def __getattr__(self, item):
        return getattr(self._pool, item)


class PoolType(Pool):
    r"""Multiprocessing stateless worker pool. See :class:`StatefulPoolType` for a pool with stateful workers.

    .. note::
        This class is only a stub for type annotation and documentation purposes only, and should not be used directly.
        Please refer to :meth:`safe_pool` for a user-facing API for constructing pool instances.
    """

    # Stub for non-stateful pool. Uninherited functions share the same signature as stubs for `Pool`.
    _state: int
    _processes: int

    def apply(self,
              fn: Callable[..., T], args: Iterable[Any] = (), kwds: Mapping[str, Any] = {}) -> T:
        r"""Calls ``fn`` with arguments ``args`` and keyword arguments ``kwds``, and blocks until the result is ready.

        Please refer to Python documentation on :py:meth:`multiprocessing.pool.Pool.apply` for details.
        """

    def apply_async(self,
                    func: Callable[..., T], args: Iterable[Any] = (), kwds: Mapping[str, Any] = {},
                    callback: Optional[Callable[[T], None]] = None,
                    error_callback: Optional[Callable[[BaseException], None]] = None) -> 'mp.pool.ApplyResult[T]':
        r"""Non-blocking version of :meth:`apply`.

        Please refer to Python documentation on :py:meth:`multiprocessing.pool.Pool.apply_async` for details.
        """

    def map(self,  # type: ignore[override]
            fn: Callable[[T], R], iterable: Iterable[T], chunksize: Optional[int] = None,
            *, args: Iterable[Any] = (), kwds: Mapping[str, Any] = {}) -> List[R]:
        r"""A parallel, eager, blocking equivalent of :meth:`map`, with support for additional arguments. The sequential
        equivalent is:

        .. code:: python

            list(map(lambda x: fn(x, *args, **kwds), iterable))

        Please refer to Python documentation on :py:meth:`multiprocessing.pool.Pool.map` for details.
        """

    def map_async(self,  # type: ignore[override]
                  fn: Callable[[T], R], iterable: Iterable[T], chunksize: Optional[int] = None,
                  callback: Optional[Callable[[T], None]] = None,
                  error_callback: Optional[Callable[[BaseException], None]] = None,
                  *, args: Iterable[Any] = (), kwds: Mapping[str, Any] = {}) -> 'mp.pool.ApplyResult[List[R]]':
        r"""Non-blocking version of :meth:`map`.

        Please refer to Python documentation on :py:meth:`multiprocessing.pool.Pool.map_async` for details.
        """

    def imap(self,  # type: ignore[override]
             fn: Callable[[T], R], iterable: Iterable[T], chunksize: int = 1,
             *, args: Iterable[Any] = (), kwds: Mapping[str, Any] = {}) -> Iterator[R]:
        r"""Lazy version of :meth:`map`.

        Please refer to Python documentation on :py:meth:`multiprocessing.pool.Pool.imap` for details.
        """

    def imap_unordered(self,  # type: ignore[override]
                       fn: Callable[[T], R], iterable: Iterable[T], chunksize: int = 1,
                       *, args: Iterable[Any] = (), kwds: Mapping[str, Any] = {}) -> Iterator[R]:
        r"""Similar to :meth:`imap`, but the ordering of the results are not guaranteed.

        Please refer to Python documentation on :py:meth:`multiprocessing.pool.Pool.imap_unordered` for details.
        """

    def starmap(self,  # type: ignore[override]
                fn: Callable[..., R], iterable: Iterable[Iterable[Any]], chunksize: Optional[int] = None,
                *, args: Iterable[Any] = (), kwds: Mapping[str, Any] = {}) -> List[R]:
        r"""Similar to :meth:`map`, except that the elements of ``iterable`` are expected to be iterables that are
        unpacked as arguments. The sequential equivalent is:

        .. code:: python

            list(map(lambda xs: fn(*xs, *args, **kwds), iterable))

        Please refer to Python documentation on :py:meth:`multiprocessing.pool.Pool.starmap` for details.
        """

    def starmap_async(self,  # type: ignore[override]
                      fn: Callable[..., R], iterable: Iterable[Iterable[Any]], chunksize: Optional[int] = None,
                      callback: Optional[Callable[[T], None]] = None,
                      error_callback: Optional[Callable[[BaseException], None]] = None,
                      *, args: Iterable[Any] = (), kwds: Mapping[str, Any] = {}) -> 'mp.pool.ApplyResult[List[R]]':
        r"""Non-blocking version of :meth:`starmap`.

        Please refer to Python documentation on :py:meth:`multiprocessing.pool.Pool.starmap_async` for details.
        """

    def gather(self,
               fn: Callable[[T], Iterator[R]], iterable: Iterable[T], chunksize: int = 1,
               *, args: Iterable[Any] = (), kwds: Mapping[str, Any] = {}) -> Iterator[R]:
        r"""Apply a function that returns a generator to each element in an iterable, and return an iterator over the
        concatenation of all elements produced by the generators. Order is not guaranteed across generators, but
        relative order is preserved for elements from the same generator.

        This method chops the iterable into a number of chunks which it submits to the process pool as separate tasks.
        The (approximate) size of these chunks can be specified by setting :attr:`chunksize` to a positive integer.

        The underlying implementation uses a managed queue to hold the results. The sequential equivalent is:

        .. code:: python

            itertools.chain.from_iterable(fn(x, *args, **kwds) for x in iterable)

        :param fn: The function returning generators.
        :param iterable: The iterable.
        :param chunksize: The (approximate) size of each chunk. Defaults to 1. A larger ``chunksize`` is beneficial
            for performance.
        :param args: Positional arguments to apply to the function.
        :param kwds: Keyword arguments to apply to the function.
        :return: An iterator over the concatenation of all elements produced by the generators.
        """


class StatefulPoolType(PoolType, Generic[State]):
    r"""Multiprocessing worker pool with per-worker states.

    Compared to stateless workers provided by the Python :mod:`multiprocessing` library, workers in a stateful pool
    have access to a process-local mutable state. The state is preserved throughout the lifetime of a worker process.
    All stateless pool methods are supported in a stateful pool. Please refer to :class:`PoolType` for a list of
    supported methods.

    The pool state class is set at construction (see :meth:`safe_pool`), and must be a subclass of :class:`PoolState`.
    A stateful pool with ``State`` as the state class supports using these functions as tasks:

    - An **unbound** method of ``State`` class. The unbound method will be bound to the process-local state upon
      dispatch.
    - Any other pickle-able function. These functions will not be able to access the pool state. As a precaution, an
      exception will be thrown if the first argument of the function is ``self``.

    Please refer to :class:`PoolState` for a comprehensive example.

    .. note::
        This class is only a stub for type annotation and documentation purposes only, and should not be used directly.
        Please refer to :meth:`safe_pool` for a user-facing API for constructing pool instances.
    """

    # Stub for stateful pool. Uninherited functions share the same signature as stubs for `PoolType`.

    def map(self,  # type: ignore[override]
            fn: Callable[[State, T], R], iterable: Iterable[T], chunksize: Optional[int] = None,
            *, args: Iterable[Any] = (), kwds: Mapping[str, Any] = {}) -> List[R]: ...

    def map_async(self,  # type: ignore[override]
                  fn: Callable[[State, T], R], iterable: Iterable[T], chunksize: Optional[int] = None,
                  callback: Optional[Callable[[T], None]] = None,
                  error_callback: Optional[Callable[[BaseException], None]] = None,
                  *, args: Iterable[Any] = (), kwds: Mapping[str, Any] = {}) -> 'mp.pool.ApplyResult[List[R]]': ...

    def imap(self,  # type: ignore[override]
             fn: Callable[[State, T], R], iterable: Iterable[T], chunksize: int = 1,
             *, args: Iterable[Any] = (), kwds: Mapping[str, Any] = {}) -> Iterator[R]: ...

    def imap_unordered(self,  # type: ignore[override]
                       fn: Callable[[State, T], R], iterable: Iterable[T], chunksize: int = 1,
                       *, args: Iterable[Any] = (), kwds: Mapping[str, Any] = {}) -> Iterator[R]: ...

    def gather(self,  # type: ignore[override]
               fn: Callable[[State, T], Iterator[R]], iterable: Iterable[T], chunksize: int = 1,
               *, args: Iterable[Any] = (), kwds: Mapping[str, Any] = {}) -> Iterator[R]: ...

    def get_states(self) -> List[State]:
        r"""Return the states of each pool worker. The pool state class can override the
        :meth:`PoolState.__return_state__` method to customize the returned value.

        The implementation uses the :meth:`broadcast` mechanism to retrieve states. This function is blocking.

        .. note::
            :meth:`get_states` must be called within the ``with`` block, before the pool terminates. Calling
            :meth:`get_states` while iterating over results from :meth:`imap`, :meth:`imap_unordered`, or :meth:`gather`
            is likely to result in deadlock or long wait periods.

        :return: A list of state for each worker process. Ordering of the states is arbitrary.
        """

    def broadcast(self, fn: Callable[[State], R],
                  *, args: Iterable[Any] = (), kwds: Mapping[str, Any] = {}) -> List[R]:
        r"""Call the function on each worker process and gather results. It is guaranteed that the function is called
        on each worker process exactly once.

        This function is blocking.

        :param fn: The function to call on workers. This must be an unbound method of the pool state class.
        :param args: Positional arguments to apply to the function.
        :param kwds: Keyword arguments to apply to the function.
        :return: A list of results, one from each worker process. Ordering of the results is arbitrary.
        """


@overload
def safe_pool(processes: int, *args, state_class: Type[State], init_args: Tuple[Any, ...] = (),
              closing: Optional[List[Any]] = None, suppress_exceptions: bool = False,
              **kwargs) -> StatefulPoolType[State]: ...


@overload
def safe_pool(processes: int, *args, state_class: Literal[None] = None,
              closing: Optional[List[Any]] = None, suppress_exceptions: bool = False, **kwargs) -> PoolType: ...


@contextlib.contextmanager  # type: ignore[misc]
def safe_pool(processes, *args, state_class=None, init_args=(), closing=None,
              suppress_exceptions=False, **kwargs):
    r"""A wrapper over :py:class:`multiprocessing.Pool <multiprocessing.pool.Pool>` with additional functionalities:

    - Fallback to sequential execution when ``processes == 0``.
    - Stateful processes: Functions run in the pool will have access to a mutable state class. See :class:`PoolState`
      for details.
    - Handles exceptions gracefully.
    - All pool methods support ``args`` and ``kwds``, which allows passing arguments to the called function.

    Please see :class:`PoolType` (non-stateful) and :class:`StatefulPoolType` for supported methods of the pool
    instance.

    :param processes: The number of worker processes to run. A value of 0 means sequential execution in the current
        process.
    :param state_class: The class of the pool state. This allows functions run by the pool to access a mutable
        process-local state. The ``state_class`` must be a subclass of :class:`PoolState`. Defaults to ``None``.
    :param init_args: Arguments to the initializer of the pool state. The state will be constructed with:

        .. code:: python

            state = state_class(*init_args)

    :param closing: An optional list of objects to close at exit, routines to run at exit. For each element ``obj``:

        - If it is a callable, ``obj`` is called with no arguments.
        - If it has an ``close()`` method, ``obj.close()`` is invoked.
        - Otherwise, an exception is raised before the pool is constructed.

    :param suppress_exceptions: If ``True``, exceptions raised within the lifetime of the pool are suppressed. Defaults
        to ``False``.
    :return: A context manager that can be used in a ``with`` statement.
    """
    if state_class is not None:
        if not issubclass(state_class, PoolState) or state_class is PoolState:
            raise ValueError("`state_class` must be a subclass of `flutes.PoolState`")

    if closing is not None and not isinstance(closing, list):
        raise ValueError("`closing` should either be `None` or a list")
    closing_fns = []
    for obj in (closing or []):
        if callable(obj):
            closing_fns.append(obj)
        elif hasattr(obj, "close") and callable(getattr(obj, "close")):
            closing_fns.append(obj.close)
        else:
            raise ValueError("Invalid object in `closing` list. "
                             "The object must either be a callable or has a `close` method")

    def close_fn():
        for fn in closing_fns:
            fn()

    if processes == 0:
        pool_class = DummyPool
    else:
        pool_class = PoolWrapper

    args = (processes,) + args
    if state_class is not None:
        pool = StatefulPool(pool_class, state_class, init_args, args, kwargs)
    else:
        pool = pool_class(*args, **kwargs)

    if processes == 0:
        # Don't swallow exceptions in the single-process case.
        yield pool
        close_fn()
        return

    try:
        yield pool
    except KeyboardInterrupt as e:
        from .log import log  # prevent circular import
        log("Gracefully shutting down...", "warning", force_console=True)
        log("Press Ctrl-C again to force terminate...", force_console=True, timestamp=False)
        try:
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pass
        raise e  # keyboard interrupts are always reraised
    except Exception as e:
        if suppress_exceptions:
            from .log import log  # prevent circular import
            log(traceback.format_exc(), force_console=True, timestamp=False)
        else:
            raise e
    finally:
        close_fn()
        # In Python 3.8, the interpreter hangs when the pool is not properly closed.
        pool.close()
        pool.terminate()


class MultiprocessingFileWriter(IO[Any]):
    r"""A multiprocessing file writer that allows multiple processes to write to the same file. Order is not guaranteed.

    This is very similar to :class:`flutes.log.MultiprocessingFileHandler`.
    """

    def __init__(self, path: PathType, mode: str = "a"):
        self._file = open(path, mode)
        self._queue: 'mp.Queue[str]' = mp.Queue(-1)

        self._thread = threading.Thread(target=self._receive)
        self._thread.daemon = True
        self._thread.start()

    def __enter__(self) -> IO[Any]:
        return self._file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._thread.join()
        self._file.close()

    def write(self, s: str):
        self._queue.put_nowait(s)

    def _receive(self):
        while True:
            try:
                record = self._queue.get()
                self._file.write(record)
            except EOFError:
                break


def kill_proc_tree(pid: int, including_parent: bool = True) -> None:
    r"""Kill all child processes of a given process.

    :param pid: The process ID (PID) of the process whose children we want to kill. To commit suicide, use
        :py:meth:`os.getpid`.
    :param including_parent: If ``True``, the process itself is killed as well. Defaults to ``True``.
    """
    import psutil
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.kill()
    _ = psutil.wait_procs(children, timeout=5)
    if including_parent:
        parent.kill()
        parent.wait(5)


class NewEvent(NamedTuple):
    worker_id: Optional[int]
    kwargs: Dict[str, Any]


class UpdateEvent(NamedTuple):
    worker_id: Optional[int]
    n: int
    postfix: Optional[Dict[str, Any]]


class WriteEvent(NamedTuple):
    worker_id: Optional[int]
    message: str


class CloseEvent(NamedTuple):
    worker_id: Optional[int]


class QuitEvent(NamedTuple):
    pass


Event = Union[NewEvent, UpdateEvent, WriteEvent, CloseEvent, QuitEvent]


class ProgressBarManager:
    r"""A manager for `tqdm <https://tqdm.github.io/>`_ progress bars that allows maintaining multiple bars from
    multiple worker processes.

    .. code:: python

        def run(xs: List[int], *, bar) -> int:
            # Create a new progress bar for the current worker.
            bar.new(total=len(xs), desc="Worker {flutes.get_worker_id()}")
            # Compute-intensive stuff!
            result = 0
            for idx, x in enumerate(xs):
                result += x
                time.sleep(random.uniform(0.01, 0.2))
                bar.update(1, postfix={"sum": result})  # update progress
                if (idx + 1) % 100 == 0:
                    # Logging works without messing up terminal output.
                    flutes.log(f"Processed {idx + 1} samples")
            return result

        def run2(xs: List[int], *, bar) -> int:
            # An alternative way to achieve the same functionalities (though slightly slower):
            result = 0
            for idx, x in enumerate(bar.iter(xs)):
                result += x
                time.sleep(random.uniform(0.01, 0.2))
                bar.update(postfix={"sum": result})  # update progress
                if (idx + 1) % 100 == 0:
                    # Logging works without messing up terminal output.
                    flutes.log(f"Processed {idx + 1} samples")
            return result

        manager = flutes.ProgressBarManager()
        # Worker processes interact with the manager through proxies.
        run_fn = functools.partial(run, bar=manager.proxy)
        with flutes.safe_pool(4) as pool:
            for idx, _ in enumerate(pool.imap_unordered(run_fn, data)):
                flutes.log(f"Processed {idx + 1} arrays")

    :param verbose: If ``False``, all progress bars are disabled. Defaults to ``True``.
    :param kwargs: Default arguments for the `tqdm <https://tqdm.github.io/>`_ progress bar initializer.
    """

    class Proxy:
        r"""Proxy class for the progress bar manager. Subprocesses should communicate with the progress bar manager
        through this class.
        """

        def __init__(self, queue: 'mp.Queue[Event]'):
            self.queue = queue

        @overload
        def new(self, iterable: Iterable[T], update_frequency: Union[int, float] = 1, **kwargs) -> Iterator[T]:
            ...

        @overload
        def new(self, iterable: Literal[None] = None, update_frequency: Union[int, float] = 1, **kwargs) -> tqdm:
            ...

        def new(self, iterable=None, update_frequency=1, **kwargs):
            r"""Construct a new progress bar.
            
            :param iterable: The iterable to decorate with a progress bar. If ``None``, then updates must be manually
                managed with calls to :meth:`update`.
            :param update_frequency: How many iterations per update. This argument only takes effect if :attr:`iterable`
                is not ``None``:

                - If :attr:`update_frequency` is a ``float``, then the progress bar is updated whenever the iterable
                  progresses over that percentage of elements. For instance, a value of ``0.01`` results in an update
                  per 1% of progress. Requires a sized iterable (having a valid ``__len__``).
                - If :attr:`update_frequency` is an ``int``, then the progress bar is updated whenever the iterable
                  progresses over that many elements. For instance, a value of ``10`` results in an update per 10
                  elements.
            :param kwargs: Additional arguments for the `tqdm <https://tqdm.github.io/>`_ progress bar initializer.
                These can override the default arguments set in the constructor of :class:`ProgressBarManager`.
            :return: The wrapped iterable, or the proxy class itself.
            """
            length = kwargs.get("total", None)
            ret_val = self
            if iterable is not None:
                try:
                    iter_len = len(iterable)
                    if length is None:
                        length = iter_len
                        kwargs.update(total=iter_len)
                    elif length != iter_len:
                        import warnings
                        warnings.warn(f"Iterable has length {iter_len} but total={length} is specified")
                except TypeError:
                    pass
                if isinstance(update_frequency, float):
                    if length is None:
                        raise ValueError("`iterable` must have valid length, or `total` must be specified "
                                         "if `update_frequency` is float")
                    if not (0.0 < update_frequency <= 1.0):
                        raise ValueError("`update_frequency` must be within the range (0, 1]")
                    ret_val = self._iter_per_percentage(iterable, length, update_frequency)
                else:
                    if not (0 < update_frequency):
                        raise ValueError("`update_frequency` must be positive")
                    ret_val = self._iter_per_elems(iterable, update_frequency)
            self.queue.put_nowait(NewEvent(get_worker_id(), kwargs))
            return ret_val

        def _iter_per_elems(self, iterable: Iterable[T], update_frequency: int) -> Iterator[T]:
            prev_index = -1
            next_index = update_frequency - 1
            idx = 0
            for idx, x in enumerate(iterable):
                yield x
                if idx == next_index:
                    self.update(idx - prev_index)
                    next_index += update_frequency
                    prev_index = idx
            # At the end, `idx == len(iterable) - 1`.
            if idx > prev_index:
                self.update(idx - prev_index)

        def _iter_per_percentage(self, iterable: Iterable[T], length: int, update_frequency: float) -> Iterator[T]:
            update_count = 0
            prev_index = -1
            next_index = max(0, int(update_frequency * length) - 1)
            for idx, x in enumerate(iterable):
                yield x
                if idx == next_index:
                    self.update(idx - prev_index)
                    update_count += 1
                    next_index = max(idx + 1, int(update_frequency * (update_count + 1) * length) - 1)
                    prev_index = idx
            if length > prev_index + 1:
                self.update(length - prev_index - 1)

        def update(self, n: int = 0, *, postfix: Optional[Dict[str, Any]] = None) -> None:
            r"""Update progress for the current progress bar.

            :param n: Increment to add to the counter.
            :param postfix: An optional dictionary containing additional stats displayed at the end of the progress bar.
                See `tqdm.set_postfix <https://tqdm.github.io/docs/tqdm/#set_postfix>`_ for more details.
            """
            self.queue.put_nowait(UpdateEvent(get_worker_id(), n, postfix))

        def write(self, message: str) -> None:
            r"""Write a message to console without disrupting the progress bars.

            :param message: The message to write.
            """
            self.queue.put_nowait(WriteEvent(get_worker_id(), message))

        def close(self) -> None:
            r"""Close the current progress bar."""
            self.queue.put_nowait(CloseEvent(get_worker_id()))

        # Methods to imitate a normal progress bar.
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()

    class _DummyProxy(Proxy):
        def __init__(self):
            pass

        def new(self, iterable=None, **kwargs):
            if iterable is not None:
                return iterable
            return self

        def update(self, n: int = 0, *, postfix: Optional[Dict[str, Any]] = None) -> None:
            pass

        def write(self, message: str) -> None:
            pass

        def close(self) -> None:
            pass

    def __init__(self, verbose: bool = True, **kwargs):
        self.verbose = verbose
        if not verbose:
            self._proxy: 'ProgressBarManager.Proxy' = self._DummyProxy()
            return

        self.manager = mp.Manager()
        self.queue: 'mp.Queue[Event]' = self.manager.Queue(-1)  # type: ignore[assignment]
        self.progress_bars: Dict[Optional[int], tqdm] = {}
        self.worker_id_map: Dict[Optional[int], int] = defaultdict(lambda: len(self.worker_id_map))
        self.bar_kwargs = kwargs.copy()
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        self._proxy = self.Proxy(self.queue)

        from .log import set_console_logging_function, _get_console_logging_function
        self._original_console_logging_fn = _get_console_logging_function()
        set_console_logging_function(self.proxy.write)

    @property
    def proxy(self):
        r"""Return the proxy class for the progress bar manager. Subprocesses should communicate with the manager
        through the proxy class.
        """
        return self._proxy

    def _close_bar(self, worker_id: Optional[int]) -> None:
        if worker_id in self.progress_bars:
            self.progress_bars[worker_id].close()
            del self.progress_bars[worker_id]

    def _run(self):
        from tqdm import tqdm
        while True:
            try:
                event = self.queue.get()
                if isinstance(event, NewEvent):
                    position = self.worker_id_map[event.worker_id]
                    self._close_bar(event.worker_id)
                    kwargs = {**self.bar_kwargs, **event.kwargs, "leave": False, "position": position}
                    bar = tqdm(**kwargs)
                    self.progress_bars[event.worker_id] = bar
                elif isinstance(event, UpdateEvent):
                    bar = self.progress_bars[event.worker_id]
                    if event.postfix is not None:
                        # Only force refresh if we're only setting the postfix.
                        bar.set_postfix(event.postfix, refresh=event.n == 0)
                    bar.update(event.n)
                elif isinstance(event, WriteEvent):
                    tqdm.write(event.message)
                elif isinstance(event, CloseEvent):
                    self._close_bar(event.worker_id)
                elif isinstance(event, QuitEvent):
                    break
                else:
                    assert False
            except (KeyboardInterrupt, SystemExit):
                raise
            except (EOFError, BrokenPipeError):
                break
            except:
                traceback.print_exc(file=sys.stderr)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if not self.verbose:
            return

        self.queue.put_nowait(QuitEvent())
        self.thread.join()
        for bar in self.progress_bars.values():
            bar.close()
        self.manager.shutdown()
        from .log import set_console_logging_function
        set_console_logging_function(self._original_console_logging_fn)
