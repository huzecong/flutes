import tempfile
from urllib.request import urlopen

import flutes


def test_reverse_open():
    url = "https://www.ltg.ed.ac.uk/~richard/unicode-sample-3-2.html"
    with tempfile.NamedTemporaryFile("wb") as f_temp:
        data = urlopen(url).read()
        f_temp.write(data)
        f_temp.flush()
        with flutes.FileProgress(open(f_temp.name)) as f:
            gold_lines = [line for line in f]
        assert flutes.get_file_lines(f_temp.name) == len(gold_lines)
        with flutes.reverse_open(f_temp.name, buf_size=10) as f:
            lines = [line for line in f]
        lines = list(reversed(lines))
    assert len(gold_lines) == len(lines)
    for gold, line in zip(gold_lines, lines):
        assert isinstance(line, str)
        assert gold == line
