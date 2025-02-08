from types import ModuleType
from typing import cast

import numpy as np
import pytest

try:
    # noinspection PyProtectedMember
    from numcodecs.zfpy import ZFPY, _zfpy  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    pytest.skip("ZFPY not available", allow_module_level=True)


from numcodecs.tests.common import (
    check_backwards_compatibility,
    check_config,
    check_encode_decode_array,
    check_err_decode_object_buffer,
    check_err_encode_object_buffer,
    check_repr,
)

_zfpy = cast(ModuleType, _zfpy)


codecs = [
    ZFPY(mode=_zfpy.mode_fixed_rate, rate=-1),
    ZFPY(),
    ZFPY(mode=_zfpy.mode_fixed_accuracy, tolerance=-1),
    ZFPY(mode=_zfpy.mode_fixed_precision, precision=-1),
]


# mix of dtypes: integer, float, bool, string
# mix of shapes: 1D, 2D, 3D
# mix of orders: C, F
arrays = [
    np.linspace(1000, 1001, 1000, dtype="f4"),
    np.linspace(1000, 1001, 1000, dtype="f8"),
    np.random.normal(loc=1000, scale=1, size=(100, 10)),
    np.random.normal(loc=1000, scale=1, size=(10, 10, 10)),
    np.random.normal(loc=1000, scale=1, size=(2, 5, 10, 10)),
    np.random.randint(-(2**31), -(2**31) + 20, size=1000, dtype="i4").reshape(100, 10),
    np.random.randint(-(2**63), -(2**63) + 20, size=1000, dtype="i8").reshape(10, 10, 10),
]


def test_encode_decode() -> None:
    for arr in arrays:
        if arr.dtype in (np.int32, np.int64):
            codec = [codecs[-1]]
        else:
            codec = codecs
        for code in codec:
            check_encode_decode_array(arr, code)


def test_config() -> None:
    for codec in codecs:
        check_config(codec)


def test_repr() -> None:
    check_repr("ZFPY(mode=4, tolerance=0.001, rate=-1, precision=-1)")


def test_backwards_compatibility() -> None:
    for code in codecs:
        if code.mode == _zfpy.mode_fixed_rate:
            codec = [code]
            check_backwards_compatibility(ZFPY.codec_id, arrays, codec)
        else:
            check_backwards_compatibility(ZFPY.codec_id, arrays[: len(arrays) - 2], codecs)


def test_err_decode_object_buffer() -> None:
    check_err_decode_object_buffer(ZFPY())


def test_err_encode_object_buffer() -> None:
    check_err_encode_object_buffer(ZFPY())


def test_err_encode_list() -> None:
    data = ['foo', 'bar', 'baz']
    for codec in codecs:
        with pytest.raises(TypeError):
            codec.encode(data)  # type: ignore[arg-type]


def test_err_encode_non_contiguous() -> None:
    # non-contiguous memory
    arr = np.arange(1000, dtype='i4')[::2]
    for codec in codecs:
        with pytest.raises(ValueError):
            codec.encode(arr)


def test_err_encode_fortran_array() -> None:
    # fortran array
    arr = np.asfortranarray(np.random.normal(loc=1000, scale=1, size=(5, 10, 20)))
    for codec in codecs:
        with pytest.raises(ValueError):
            codec.encode(arr)
