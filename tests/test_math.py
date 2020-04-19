import flutes


def test_ceildiv() -> None:
    assert flutes.ceil_div(5, 5) == 1
    assert flutes.ceil_div(3, 4) == 1
    assert flutes.ceil_div(6, 5) == 2
    assert flutes.ceil_div(8, 1) == 8
