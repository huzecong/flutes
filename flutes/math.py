__all__ = [
    "ceil_div",
]


def ceil_div(a: int, b: int) -> int:
    r"""Integer division that rounds up."""
    return (a - 1) // b + 1
