import subprocess

import pytest

import flutes


def test_run_command() -> None:
    with open(__file__, "rb") as f:
        code = f.read()
    result = flutes.run_command(["cat", __file__], verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == code

    with pytest.raises(subprocess.CalledProcessError, match=r"Captured output:\n\s+Test output"):
        flutes.run_command(["sh", "-c", "echo 'Test output'; exit 1"], verbose=True)
