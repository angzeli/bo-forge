"""Runtime call boundaries, deliberately independent of annotation spelling."""

import inspect


def call_shape(value):
    signature = inspect.signature(value)
    return [
        {
            "name": parameter.name,
            "kind": parameter.kind.name,
            "default": None if parameter.default is inspect.Parameter.empty
            else repr(parameter.default),
            "required": parameter.default is inspect.Parameter.empty,
        }
        for parameter in signature.parameters.values()
    ]


def public_calls():
    import bo_forge

    result = {}
    for name in bo_forge.__all__:
        value = getattr(bo_forge, name)
        if not callable(value):
            continue
        try:
            result[name] = call_shape(value)
        except ValueError:
            # Built-in Exception constructors have no inspectable Python signature.
            assert isinstance(value, type) and issubclass(value, Exception)
        if isinstance(value, type):
            for member_name, descriptor in vars(value).items():
                if member_name.startswith("_"):
                    continue
                member = descriptor.fget if isinstance(descriptor, property) else getattr(
                    value, member_name,
                )
                if callable(member):
                    result[f"{name}.{member_name}"] = call_shape(member)
    return result
