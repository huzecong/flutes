import tempfile
from urllib.request import urlopen

import flutes

URL = "https://www.ltg.ed.ac.uk/~richard/unicode-sample-3-2.html"
DATA = urlopen(URL).read()


def test_reverse_open() -> None:
    with tempfile.NamedTemporaryFile("wb") as f_temp:
        f_temp.write(DATA)
        f_temp.flush()
        with open(f_temp.name) as f:
            gold_lines = [line for line in f]
        assert flutes.get_file_lines(f_temp.name) == len(gold_lines)
        with flutes.reverse_open(f_temp.name, buffer_size=10) as f:
            lines = [line for line in f]
        lines = list(reversed(lines))
    assert len(gold_lines) == len(lines)
    for gold, line in zip(gold_lines, lines):
        assert isinstance(line, str)
        assert gold == line


@flutes.shut_up(stderr=True)
def test_progress_open() -> None:
    def _test(modes=None):
        def decorator(func):
            def wrapped():
                for verbose in [False, True]:
                    for mode in (modes or ["r", "rb"]):
                        with flutes.progress_open(f_temp.name, mode, verbose=verbose) as f:
                            func(f)
                            if verbose:
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

        test_iter_line()
        test_seek()
