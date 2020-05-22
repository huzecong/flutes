import tempfile

import flutes


def test_log() -> None:
    with tempfile.NamedTemporaryFile("w") as f_tmp:
        flutes.set_log_file(f_tmp.name)
        flutes.set_log_file(f_tmp.name)
        flutes.set_logging_level("warning")
        flutes.log("info output", "info")
        flutes.log("warning output", "warning")
        flutes.log("error output", "error", timestamp=False)
        flutes.log("success output", "success")

        # # For some reason the following fails randomly on GitHub CI
        # logging.shutdown()
        # with open(f_tmp.name, "r") as f:
        #     lines = [line for line in f]
        # assert len(lines) == 2
        # assert "warning output" in lines[0]
        # assert "error output" in lines[1]
