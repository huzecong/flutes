import flutes


def test_exception_wrapper():
    def handler_fn(e, three, one, args, my_arg=None, **kw):
        assert isinstance(e, ValueError)
        assert str(e) == "test"
        assert three is None
        assert one == 1
        assert args == ("arg1", "arg2")
        assert my_arg is None
        assert kw == {"two": "2",
                      "kwargs": {"four": 4}}

    @flutes.exception_wrapper(handler_fn)
    def foo(one, two, *args, three=None, **kwargs):
        raise ValueError("test")

    foo(1, "2", "arg1", "arg2", four=4)

    @flutes.exception_wrapper()
    def foo():
        raise ValueError("test2")

    foo()
