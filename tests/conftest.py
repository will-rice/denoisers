"""Pytest configuration and fixtures."""

import builtins
from functools import wraps

import numpy as np
from packaging.version import parse as _parse_version

# Fix pyroomacoustics incompatibility with numpy 2.x.
# In numpy 2.x, float() on a 1D single-element array raises TypeError.
# pyroomacoustics.room.Room.extrude calls float() on wall.absorption (a 1D array).
if _parse_version(np.__version__) >= _parse_version("2.0"):
    try:
        import pyroomacoustics.room as _pra_room

        _orig_extrude = _pra_room.Room.extrude
        _orig_float = builtins.float

        @wraps(_orig_extrude)
        def _compat_extrude(self, *args, **kwargs):
            """Wrapper for Room.extrude with numpy 2.x compatibility."""

            class _Float(_orig_float):
                def __new__(cls, x):
                    if isinstance(x, np.ndarray) and x.ndim > 0 and x.size == 1:
                        return _orig_float.__new__(cls, x.flat[0])
                    return _orig_float.__new__(cls, x)

            builtins.float = _Float
            try:
                return _orig_extrude(self, *args, **kwargs)
            finally:
                builtins.float = _orig_float

        _pra_room.Room.extrude = _compat_extrude
    except (ImportError, AttributeError):
        pass
