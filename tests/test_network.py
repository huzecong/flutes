import json
import os
import tempfile
from pathlib import Path

import flutes


def test_download() -> None:
    urls = [
        ("https://drive.google.com/file/d/1bUShb-0taoXGDveut1B31UqzR-M7fEwA/view?usp=sharing", "demo.txt"),
        ("https://github.com/Somefive/MercuryJson/raw/master/data/numbers-small.json", "numbers-small.json"),
    ]

    with tempfile.TemporaryDirectory() as tempdir:
        paths = []
        for url, filename in urls:
            path = Path(flutes.download(url, save_dir=tempdir, filename=filename))
            assert path.name == filename
            assert path.parent == Path(tempdir)
            paths.append(path)

        with paths[0].open() as f:
            assert f.read().strip() == "This is a demo file from Google Drive."

        with paths[1].open() as f:
            assert all(isinstance(x, float) for x in json.load(f))

    for url, filename in urls[1:]:
        download_path = Path(tempfile.gettempdir()) / filename
        if download_path.exists():
            download_path.unlink()
        assert flutes.download(url, progress=True) == str(download_path)
