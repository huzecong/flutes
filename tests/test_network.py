import json
import tempfile
from pathlib import Path

import flutes


def test_download() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(flutes.download(
            "https://drive.google.com/file/d/1bUShb-0taoXGDveut1B31UqzR-M7fEwA/view?usp=sharing",
            save_dir=tempdir, filename="demo.txt"))
        assert path.name == "demo.txt"
        assert path.parent == Path(tempdir)
        with path.open() as f:
            assert f.read().strip() == "This is a demo file from Google Drive."

        path = Path(flutes.download(
            "https://github.com/Somefive/MercuryJson/raw/master/data/numbers-small.json", save_dir=tempdir))
        assert path.name == "numbers-small.json"
        assert path.parent == Path(tempdir)
        with path.open() as f:
            assert all(isinstance(x, float) for x in json.load(f))
