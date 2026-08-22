"""Small local EasyDict-compatible implementation for offline environments."""
from __future__ import annotations

from collections.abc import Mapping


def _convert(value):
    if isinstance(value, Mapping) and not isinstance(value, EasyDict):
        return EasyDict(value)
    if isinstance(value, list):
        return [_convert(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_convert(item) for item in value)
    return value


class EasyDict(dict):
    """Dictionary supporting recursive attribute-style access."""

    def __init__(self, mapping=None, **kwargs):
        source = {}
        if mapping is not None:
            source.update(dict(mapping))
        source.update(kwargs)
        super().__init__((key, _convert(value)) for key, value in source.items())

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = _convert(value)

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def update(self, *args, **kwargs):
        values = dict(*args, **kwargs)
        for key, value in values.items():
            self[key] = _convert(value)
