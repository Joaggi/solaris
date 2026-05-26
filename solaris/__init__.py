"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

__all__ = [
    "Solaris",
    "SolarisHighRes",
    "SolarisSmall",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from solaris.model.solaris import Solaris, SolarisHighRes, SolarisSmall

    return {
        "Solaris": Solaris,
        "SolarisHighRes": SolarisHighRes,
        "SolarisSmall": SolarisSmall,
    }[name]
