from functools import lru_cache
from typing import Callable, Collection, Dict, List, Sequence, Set, Type, TypeVar, no_type_check

__all__ = [
    "reverse_map",
    "register_no_map_class",
    "no_map_instance",
    "map_structure",
    "map_structure_zip",
]

T = TypeVar('T')
R = TypeVar('R')


def reverse_map(d: Dict[T, int]) -> List[T]:
    r"""Given a dict containing pairs of ``(item, id)``, return a list where the ``id``-th element is ``item``.

    .. note::
        It is assumed that the ``id``\ s form a permutation.

    .. code:: python

        >>> words = ['a', 'aardvark', 'abandon', ...]
        >>> word_to_id = {word: idx for idx, word in enumerate(words)}
        >>> id_to_word = reverse_map(word_to_id)
        >>> (words == id_to_word)
        True

    :param d: The dictionary mapping ``item`` to ``id``.
    """
    return [k for k, _ in sorted(d.items(), key=lambda xs: xs[1])]


_NO_MAP_TYPES: Set[type] = set()
_NO_MAP_INSTANCE_ATTR = "--no-map--"


def register_no_map_class(container_type: Type[T]) -> None:
    r"""Register a container type as `non-mappable`, i.e., instances of the class will be treated as singleton objects in
    :func:`map_structure` and :func:`map_structure_zip`, their contents will not be traversed. This would be useful for
    certain types that subclass built-in container types, such as ``torch.Size``.

    :param container_type: The type of the container, e.g. :py:class:`list`, :py:class:`dict`.
    """
    return _NO_MAP_TYPES.add(container_type)


@lru_cache(maxsize=None)
def _no_map_type(container_type: Type[T]) -> Type[T]:
    # Create a subtype of the container type that sets an normally inaccessible
    # special attribute on instances.
    # This is necessary because `setattr` does not work on built-in types
    # (e.g. `list`).
    new_type = type("_no_map" + container_type.__name__,
                    (container_type,), {_NO_MAP_INSTANCE_ATTR: True})
    return new_type


@no_type_check
def no_map_instance(instance: T) -> T:
    r"""Register a container instance as `non-mappable`, i.e., it will be treated as a singleton object in
    :func:`map_structure` and :func:`map_structure_zip`, its contents will not be traversed.

    :param instance: The container instance.
    """
    try:
        setattr(instance, _NO_MAP_INSTANCE_ATTR, True)
        return instance
    except AttributeError:
        return _no_map_type(type(instance))(instance)


@no_type_check
def map_structure(fn: Callable[[T], R], obj: Collection[T]) -> Collection[R]:
    r"""Map a function over all elements in a (possibly nested) collection.

    :param fn: The function to call on elements.
    :param obj: The collection to map function over.
    :return: The collection in the same structure, with elements mapped.
    """
    if obj.__class__ in _NO_MAP_TYPES or hasattr(obj, _NO_MAP_INSTANCE_ATTR):
        return fn(obj)
    if isinstance(obj, list):
        return [map_structure(fn, x) for x in obj]
    if isinstance(obj, tuple):
        if hasattr(obj, '_fields'):  # namedtuple
            return type(obj)(*[map_structure(fn, x) for x in obj])
        else:
            return tuple(map_structure(fn, x) for x in obj)
    if isinstance(obj, dict):
        # could be `OrderedDict`
        return type(obj)((k, map_structure(fn, v)) for k, v in obj.items())
    if isinstance(obj, set):
        return {map_structure(fn, x) for x in obj}
    return fn(obj)


@no_type_check
def map_structure_zip(fn: Callable[..., R], objs: Sequence[Collection[T]]) -> Collection[R]:
    r"""Map a function over tuples formed by taking one elements from each (possibly nested) collection. Each collection
    must have identical structures.

    .. note::
        Although identical structures are required, it is not enforced by assertions. The structure of the first
        collection is assumed to be the structure for all collections.

    :param fn: The function to call on elements.
    :param objs: The list of collections to map function over.
    :return: A collection with the same structure, with elements mapped.
    """
    obj = objs[0]
    if obj.__class__ in _NO_MAP_TYPES or hasattr(obj, _NO_MAP_INSTANCE_ATTR):
        return fn(*objs)
    if isinstance(obj, list):
        return [map_structure_zip(fn, xs) for xs in zip(*objs)]
    if isinstance(obj, tuple):
        if hasattr(obj, '_fields'):  # namedtuple
            return type(obj)(*[map_structure_zip(fn, xs) for xs in zip(*objs)])
        else:
            return tuple(map_structure_zip(fn, xs) for xs in zip(*objs))
    if isinstance(obj, dict):
        # could be `OrderedDict`
        return type(obj)((k, map_structure_zip(fn, [o[k] for o in objs])) for k in obj.keys())
    if isinstance(obj, set):
        raise ValueError("Structures cannot contain `set` because it's unordered")
    return fn(*objs)
