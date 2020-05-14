import contextlib
import functools
import inspect
import multiprocessing as mp
import sys
import threading
import traceback
import types
from collections import defaultdict
from multiprocessing.pool import Pool
from types import FrameType
from typing import (Any, Callable, Dict, Generic, IO, Iterable, Iterator, List, NamedTuple, Optional, Set, Tuple, Type,
                    TypeVar, Union, cast, no_type_check, overload)

from tqdm import tqdm
from typing_extensions import Literal

from .types import PathType

__all__ = [
    "get_worker_id",
    "PoolState",
    "safe_pool",
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

    def imap(self, fn: Callable[[T], R], iterable: Iterable[T], *_, **__) -> Iterator[R]:
        if self._process_state is not None:
            locals().update({"__state__": self._process_state})
        yield from map(fn, iterable)

    def imap_unordered(self, fn: Callable[[T], R], iterable: Iterable[T], *_, **__) -> Iterator[R]:
        if self._process_state is not None:
            locals().update({"__state__": self._process_state})
        yield from map(fn, iterable)

    def map(self, fn: Callable[[T], R], iterable: Iterable[T], *_, **__) -> List[R]:
        if self._process_state is not None:
            locals().update({"__state__": self._process_state})
        return [fn(x) for x in iterable]

    def map_async(self, fn: Callable[[T], R], iterable: Iterable[T], *_, **__) -> 'mp.pool.ApplyResult[List[R]]':
        if self._process_state is not None:
            locals().update({"__state__": self._process_state})
        return DummyApplyResult(self.map(fn, iterable))

    def starmap(self, fn: Callable[..., R], iterable: Iterable[Tuple[T, ...]], *_, **__) -> List[R]:
        if self._process_state is not None:
            locals().update({"__state__": self._process_state})
        return [fn(*x) for x in iterable]

    def starmap_async(self, fn: Callable[..., R], iterable: Iterable[Tuple[T, ...]], *_, **__) \
            -> 'mp.pool.ApplyResult[List[R]]':
        if self._process_state is not None:
            locals().update({"__state__": self._process_state})
        return DummyApplyResult(self.starmap(fn, iterable))

    def apply(self, fn: Callable[..., R], args: Iterable[Any] = (), kwds: Dict[str, Any] = {}, *_, **__) -> R:
        if self._process_state is not None:
            locals().update({"__state__": self._process_state})
        return fn(*args, **kwds)

    def apply_async(self, fn: Callable[..., R], args: Iterable[Any] = (), kwds: Dict[str, Any] = {}, *_, **__) \
            -> 'mp.pool.ApplyResult[R]':
        if self._process_state is not None:
            locals().update({"__state__": self._process_state})
        return DummyApplyResult(self.apply(fn, args, kwds))

    @staticmethod
    def _no_op(self, *args, **kwargs):
        pass

    def __getattr__(self, item):
        return types.MethodType(DummyPool._no_op, self)  # no-op for everything else

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._state = mp.pool.TERMINATE


PoolType = Union[Pool, DummyPool]


def get_worker_id() -> Optional[int]:
    r"""Return the ID of the pool worker process, or ``None`` if the current process is not a pool worker."""
    proc_name = mp.current_process().name
    if "PoolWorker" in proc_name:
        worker_id = int(proc_name[(proc_name.find('-') + 1):])
        return worker_id
    return None


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

    A stateful pool with ``State`` as the state class supports using these functions as tasks:

    - An **unbound** method of ``State`` class. The unbound method will be bound to the process-local state upon
      dispatch.
    - Any other pickle-able function. These functions will not be able to access the pool state. As a precaution, an
      exception will be thrown if the first argument of the function is ``self``.

    .. note::
        ``pool.get_state()`` is only available for stateful pools, and must be called within the ``with`` block, before
        the pool terminates. When invoked, additional tasks to retrieve the states are added to the pool's task queue,
        and the function will block until the tasks complete.

    **See also:** :func:`safe_pool`
    """
    # Dummy base class for pool processor states.

    def __return_state__(self):
        r"""When ``pool.get_states()`` is invoked, this method is called for each pool worker to return its state. The
        default implementation returns the :class:`PoolState` object itself, but it might be beneficial to override this
        method in certain cases:

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


def _pool_fn_with_state(fn: Callable[..., R], *args, **kwargs) -> R:
    # Wrapper for compute function passed to stateful pools.
    frame = cast(FrameType, inspect.currentframe().f_back)  # type: ignore[union-attr]
    if '__state__' not in frame.f_locals:  # in DummyPool we have only one layer on the stack
        frame = cast(FrameType, frame.f_back)  # _pool_fn_with_state -> mapper -> worker
    local_vars = frame.f_locals
    state_obj = local_vars['__state__']
    del frame, local_vars
    return fn(state_obj, *args, **kwargs)


def _chain_fns(fns: List[Callable[..., R]], fn_arg_kwargs: List[Tuple[Tuple[Any, ...], Dict[str, Any]]]) -> List[R]:
    rets = []
    for fn, (args, kwargs) in zip(fns, fn_arg_kwargs):
        rets.append(fn(*args, **kwargs))
    return rets


State = TypeVar('State', bound=PoolState)


class StatefulPoolWrapper(Generic[State]):
    _pool: PoolType
    _state_class: Type[State]
    _class_methods: Set[int]  # a list of addresses of instance methods for the state class

    def __init__(self, pool_class: Type[PoolType], state_class: Type[State], state_init_args: Tuple[Any, ...],
                 args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        self._state_class = state_class

        # Store the IDs of all methods of the `PoolState` subclass.
        self._class_methods = set()
        for attr_name in dir(self._state_class):
            attr_val = getattr(self._state_class, attr_name)
            if inspect.isfunction(attr_val):
                self._class_methods.add(id(attr_val))
        self._class_methods.add(id(self._return_state))  # add special method

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

        methods = ["imap", "imap_unordered", "map", "map_async", "starmap", "starmap_async", "apply", "apply_async"]
        for name in methods:
            pool_method = getattr(self._pool, name)
            wrapped_method = self._define_method(pool_method)
            setattr(self, name, wrapped_method)

    @no_type_check
    def _wrap_fn(self, func: Callable[[State, T], R]) -> Callable[[T], R]:
        # If the function is a `PoolState` method, wrap it to allow access to `self`.
        if id(func) in self._class_methods:
            return functools.partial(_pool_fn_with_state, func)
        if inspect.ismethod(func):
            if func.__self__.__class__ is self._state_class:
                raise ValueError(f"Bound methods of the pool state class {self._state_class.__name__} are not "
                                 f"accepted; use an unbound method instead.")
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
    def _return_state(self: State, received_ids: Set[int]) -> Optional[Tuple[State, int]]:
        worker_id = get_worker_id()
        assert worker_id is not None
        if worker_id in received_ids:
            return None
        return (self.__return_state__(), worker_id)

    def get_states(self) -> List[State]:
        if self._pool._state == mp.pool.TERMINATE:  # type: ignore[union-attr]
            raise ValueError("Pool is already terminated")
        if isinstance(self._pool, DummyPool):
            return [self._pool._process_state.__return_state__()]  # type: ignore[union-attr]
        assert isinstance(self._pool, Pool)
        received_ids: Set[int] = set()
        states = []
        while len(received_ids) < self._pool._processes:  # type: ignore[attr-defined]
            result = self.apply(self._return_state, (received_ids,))
            if result is not None:
                state, worker_id = result
                received_ids.add(worker_id)
                states.append(state)
        return states

    def __getattr__(self, item):
        return getattr(self._pool, item)


class StatefulPoolType(Pool, Generic[State]):
    # Stub for stateful pool. Uninherited functions share the same signature as stubs for `Pool`.

    def imap(self,  # type: ignore[override]
             fn: Callable[[State, T], R], iterable: Iterable[T], chunksize: int = 1) -> Iterator[R]: ...

    def imap_unordered(self,  # type: ignore[override]
                       fn: Callable[[State, T], R], iterable: Iterable[T], chunksize: int = 1) -> Iterator[R]: ...

    def map(self,  # type: ignore[override]
            fn: Callable[[State, T], R], iterable: Iterable[T], chunksize: Optional[int] = None) -> List[R]: ...

    def map_async(self,  # type: ignore[override]
                  fn: Callable[[State, T], R], iterable: Iterable[T], chunksize: Optional[int] = None,
                  callback: Optional[Callable[[T], None]] = None,
                  error_callback: Optional[Callable[[BaseException], None]] = None) \
            -> 'mp.pool.ApplyResult[List[R]]': ...

    def get_states(self) -> List[State]: ...


@overload
def safe_pool(processes: int, *args, state_class: Type[State], init_args: Tuple[Any, ...] = (),
              closing: Optional[List[Any]] = None, **kwargs) -> StatefulPoolType[State]: ...


@overload
def safe_pool(processes: int, *args, state_class: Literal[None] = None,
              closing: Optional[List[Any]] = None, **kwargs) -> Pool: ...


@contextlib.contextmanager  # type: ignore[misc]
def safe_pool(processes, *args, state_class=None, init_args=(), closing=None, **kwargs):
    r"""A wrapper over :py:class:`multiprocessing.Pool <multiprocessing.pool.Pool>` with additional functionalities:

    - Fallback to sequential execution when ``processes == 0``.
    - Stateful processes: Functions run in the pool will have access to a mutable state class. See :class:`PoolState`
      for details.
    - Handles exceptions gracefully.

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
        - Otherwise, it is ignored.

    :return: A context manager that can be used in a ``with`` statement.
    """
    if state_class is not None:
        if not issubclass(state_class, PoolState) or state_class is PoolState:
            raise ValueError("`state_class` must be a subclass of `flutes.PoolState`")

    if closing is not None and not isinstance(closing, list):
        raise ValueError("`closing` should either be `None` or a list")

    def close_fn():
        for obj in (closing or []):
            if callable(obj):
                obj()
            elif hasattr(obj, "close") and callable(getattr(obj, "close")):
                obj.close()

    if processes == 0:
        pool_class = DummyPool
    else:
        pool_class = mp.Pool

    args = (processes,) + args
    if state_class is not None:
        pool = StatefulPoolWrapper(pool_class, state_class, init_args, args, kwargs)
    else:
        pool = pool_class(*args, **kwargs)

    if processes == 0:
        # Don't swallow exceptions in the single-process case.
        yield pool
        close_fn()
        return

    try:
        yield pool
    except KeyboardInterrupt:
        from .log import log  # prevent circular import
        log("Gracefully shutting down...", "warning", force_console=True)
        print("Press Ctrl-C again to force terminate...")
        try:
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pass
    except Exception:
        print(traceback.format_exc())
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

    :param pid: The process ID (PID) of the process whose children we want to kill.
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


class QuitEvent(NamedTuple):
    pass


Event = Union[NewEvent, UpdateEvent, WriteEvent, QuitEvent]


class ProgressBarManager:
    r"""A manager for `tqdm <https://tqdm.github.io/>`_ progress bars that allows maintaining multiple bars from
    multiple worker processes.

    .. code:: python

        def run(xs: List[int], *, bar) -> int:
            bar.new(total=len(xs), desc="Worker {flutes.get_worker_id()}")  # create a new progress bar
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
        run_fn = functools.partial(run, bar=manager.proxy)
        with flutes.safe_pool(4) as pool:
            for idx, _ in enumerate(pool.imap_unordered(run_fn, data)):
                flutes.log(f"Processed {idx + 1} arrays")

    :param kwargs: Default arguments for the `tqdm <https://tqdm.github.io/>`_ progress bar initializer.
    """

    class Proxy:
        def __init__(self, queue: 'mp.Queue[Event]'):
            self.queue = queue

        @overload
        def new(self, iterable: Iterable[T], **kwargs) -> Iterator[T]:
            ...

        @overload
        def new(self, iterable: Literal[None] = None, **kwargs) -> tqdm:
            ...

        def new(self, iterable=None, **kwargs):
            r"""Construct a new progress bar."""
            if iterable is not None:
                length = None
                try:
                    length = len(iterable)
                    kwargs.update(total=length)
                except TypeError:
                    pass
            self.queue.put_nowait(NewEvent(get_worker_id(), kwargs))
            if iterable is not None:
                return self._iter(iterable)
            return self

        def _iter(self, iterable: Iterable[T]) -> Iterator[T]:
            for x in iterable:
                yield x
                self.update(1)

        def update(self, n: int = 0, *, postfix: Optional[Dict[str, Any]] = None) -> None:
            # TODO: Add throttle to prevent sending messages too often.
            self.queue.put_nowait(UpdateEvent(get_worker_id(), n, postfix))

        def write(self, message: str) -> None:
            self.queue.put_nowait(WriteEvent(get_worker_id(), message))

        # Methods to imitate a normal progress bar.
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def close(self):
            pass

    def __init__(self, **kwargs):
        self.manager = mp.Manager()
        self.queue: 'mp.Queue[Event]' = self.manager.Queue(-1)
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
        return self._proxy

    def _run(self):
        from tqdm import tqdm
        while True:
            try:
                event = self.queue.get()
                if isinstance(event, NewEvent):
                    position = self.worker_id_map[event.worker_id]
                    if event.worker_id in self.progress_bars:
                        self.progress_bars[event.worker_id].close()
                        del self.progress_bars[event.worker_id]
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
        self.queue.put_nowait(QuitEvent())
        self.thread.join()
        for bar in self.progress_bars.values():
            bar.close()
        from .log import set_console_logging_function
        set_console_logging_function(self._original_console_logging_fn)
