from collections.abc import Callable
from functools import partial, update_wrapper
from inspect import signature
from typing import Any, TypeVar, cast

Func = TypeVar("Func", bound=Callable)
Return = TypeVar("Return")


def format_docstring(*args: Any, **kwargs: Any) -> Callable[[Func], Func]:
    def decorator(func: Func) -> Func:
        assert func.__doc__ is not None
        func.__doc__ = func.__doc__.format(*args, **kwargs)
        return func

    return decorator


def partial_with_doc(
    tool: Callable[..., Return], **kwargs: Any
) -> Callable[..., Return]:
    wrapped = partial(tool, **kwargs)
    update_wrapper(wrapped, tool)
    tool_signature = signature(tool)
    setattr(
        wrapped,
        "__signature__",
        tool_signature.replace(
            parameters=[
                parameter
                for name, parameter in tool_signature.parameters.items()
                if name not in kwargs
            ]
        ),
    )
    return cast(Callable[..., Return], wrapped)
