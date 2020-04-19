import tempfile
from urllib.request import urlopen

import flutes

URL = "https://www.ltg.ed.ac.uk/~richard/unicode-sample-3-2.html"
DATA = urlopen(URL).read()


def test_reverse_open():
    with tempfile.NamedTemporaryFile("wb") as f_temp:
        f_temp.write(DATA)
        f_temp.flush()
        with open(f_temp.name) as f:
            gold_lines = [line for line in f]
        assert flutes.get_file_lines(f_temp.name) == len(gold_lines)
        with flutes.reverse_open(f_temp.name, buf_size=10) as f:
            lines = [line for line in f]
        lines = list(reversed(lines))
    assert len(gold_lines) == len(lines)
    for gold, line in zip(gold_lines, lines):
        assert isinstance(line, str)
        assert gold == line


@flutes.shut_up(stderr=True)
def test_progress_open():
    def _test(modes=None):
        def decorator(func):
            def wrapped():
                for mode in (modes or ["r", "rb"]):
                    with flutes.progress_open(f_temp.name, mode) as f:
                        func(f)
                        assert f.progress_bar.n == f.progress_bar.total

            return wrapped

        return decorator

    with tempfile.NamedTemporaryFile("wb") as f_temp:
        for _ in range(100):
            f_temp.write(DATA)
        f_temp.flush()

        @_test()
        def test_iter_line(f):
            for line in f:
                pass

        @_test()
        def test_seek(f):
            f.seek(5000)
            f.read()

        @_test(modes=["rb"])
        def test_readinto(f):
            b = bytearray(10000)
            f.seek(5000)
            f.readinto(b)
            content = f.read()
            assert f.progress_bar.total == 5000 + len(b) + len(content)

        test_iter_line()
        test_seek()
        test_readinto()
