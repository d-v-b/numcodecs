import zlib as _zlib
from typing import ClassVar, Literal

from typing_extensions import Buffer

from numcodecs.ndarray_like import NDArrayLike

from .abc import Codec
from .compat import ensure_contiguous_ndarray, ndarray_copy


class Zlib(Codec[Literal['zlib']]):
    """Codec providing compression using zlib via the Python standard library.

    Parameters
    ----------
    level : int
        Compression level.

    """
    level: int

    def __init__(self, level: int = 1) -> None:
        self.level = level

    def encode(self, buf: Buffer) -> Buffer:
        # normalise inputs
        buf = ensure_contiguous_ndarray(buf)

        # do compression
        return _zlib.compress(buf, self.level)

    # noinspection PyMethodMayBeStatic
    def decode(self, buf: Buffer, out: Buffer | None = None) -> NDArrayLike:
        # normalise inputs
        buf = ensure_contiguous_ndarray(buf)
        if out is not None:
            out = ensure_contiguous_ndarray(out)

        # do decompression
        dec = _zlib.decompress(buf)

        # handle destination - Python standard library zlib module does not
        # support direct decompression into buffer, so we have to copy into
        # out if given
        return ndarray_copy(dec, out)
