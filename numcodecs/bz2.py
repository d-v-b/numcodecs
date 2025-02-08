from __future__ import annotations

import bz2 as _bz2
from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from typing_extensions import Buffer

from numcodecs.abc import Codec, ConfigDict
from numcodecs.compat import ensure_contiguous_ndarray, ndarray_copy

class BZ2Config(ConfigDict[Literal['bz2']]):
    level: int

class BZ2(Codec[Literal['bz2']]):
    """Codec providing compression using bzip2 via the Python standard library.

    Parameters
    ----------
    level : int
        Compression level.

    """
    level: int

    def __init__(self, level: int = 1) -> None:
        self.level = level

    def encode(self, buf: Buffer) -> bytes:
        # normalise input
        buf = ensure_contiguous_ndarray(buf)

        # do compression
        return _bz2.compress(buf, self.level)

    # noinspection PyMethodMayBeStatic
    def decode(self, buf: Buffer, out: Buffer | None = None) -> Buffer:
        # normalise inputs
        buf = ensure_contiguous_ndarray(buf)
        if out is not None:
            out = ensure_contiguous_ndarray(out)

        # N.B., bz2 cannot handle ndarray directly because of truth testing issues
        buf = memoryview(buf)

        # do decompression
        dec = _bz2.decompress(buf)

        # handle destination - Python standard library bz2 module does not
        # support direct decompression into buffer, so we have to copy into
        # out if given
        return ndarray_copy(dec, out)
