import time

import flutes


def test_work_in_progress() -> None:
    with flutes.work_in_progress("abc"):
        time.sleep(0.1)
