import os
import tempfile
from pathlib import Path

import flutes


def test_copy_tree() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir)
        result = flutes.run_command(["git", "clone", "https://github.com/huzecong/flutes"], cwd=path)
        assert result.return_code == 0
        flutes.copy_tree(path / "flutes", path / "flutes_copy")
        assert flutes.get_folder_size(path / "flutes") == flutes.get_folder_size(path / "flutes_copy")
        assert (sorted(flutes.scandir(path / "flutes")) ==
                sorted([(path / "flutes" / p).absolute() for p in os.listdir(path / "flutes")]))


def test_readable_size() -> None:
    assert flutes.readable_size(500.12345) == "500.12"
    assert flutes.readable_size(2048) == "2.00K"
    assert flutes.readable_size(34.567 * 1024 ** 5) == "34.57P"


def test_remove_prefix() -> None:
    assert flutes.remove_prefix("some string", "some ") == "string"
    assert flutes.remove_prefix("some string", "something") == " string"
    assert flutes.remove_prefix("some string", "not matching") == "some string"
    assert flutes.remove_prefix("some string", "some string longer") == ""


def test_cache() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "cache.pkl")

        @flutes.cache(path)
        def gen_obj():
            return {str(i): (i, [i, i], {i: i}) for i in range(100)}

        obj = gen_obj()
        assert os.path.exists(path)
        assert gen_obj() == obj
