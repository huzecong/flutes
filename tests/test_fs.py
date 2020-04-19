import os
import tempfile

import flutes


def test_readable_size():
    assert flutes.readable_size(500.12345) == "500.12"
    assert flutes.readable_size(2048) == "2.00K"
    assert flutes.readable_size(34.567 * 1024 ** 5) == "34.57P"


def test_remove_prefix():
    assert flutes.remove_prefix("some string", "some ") == "string"
    assert flutes.remove_prefix("some string", "something") == " string"
    assert flutes.remove_prefix("some string", "not matching") == "some string"
    assert flutes.remove_prefix("some string", "some string longer") == ""


def test_cache():
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "cache.pkl")

        @flutes.cache(path)
        def gen_obj():
            return {str(i): (i, [i, i], {i: i}) for i in range(100)}

        obj = gen_obj()
        assert os.path.exists(path)
        assert gen_obj() == obj
