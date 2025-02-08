from typing import Any, Literal

import numpy as np
from typing_extensions import Buffer

from ._shuffle import _doShuffle, _doUnshuffle  # type: ignore[import-untyped]
from .abc import Codec
from .compat import ensure_contiguous_ndarray


class Shuffle(Codec[Literal['shuffle']]):
    """Codec providing shuffle

    Parameters
    ----------
    elementsize : int
        Size in bytes of the array elements.  Default = 4

    """
    elementsize: int

    def __init__(self, elementsize: int = 4) -> None:
        self.elementsize = elementsize

    def _prepare_arrays(
        self, buf: Buffer, out: Buffer | None
    ) -> tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
        buf = ensure_contiguous_ndarray(buf)

        if out is None:
            out = np.zeros(buf.nbytes, dtype='uint8')
        else:
            out = ensure_contiguous_ndarray(out)

        if self.elementsize <= 1:
            out.view(buf.dtype)[: len(buf)] = buf[:]  # no shuffling needed
            return buf, out

        if buf.nbytes % self.elementsize != 0:
            raise ValueError("Shuffle buffer is not an integer multiple of elementsize")

        return buf, out

    def encode(self, buf: Buffer, out: Buffer | None = None) -> np.ndarray[Any, np.dtype[Any]]:
        buf, out = self._prepare_arrays(buf, out)

        if self.elementsize <= 1:
            return out  # no shuffling needed

        _doShuffle(buf.view("uint8"), out.view("uint8"), self.elementsize)

        return out

    def decode(self, buf: Buffer, out: Buffer | None = None) -> np.ndarray[Any, np.dtype[Any]]:
        buf, out = self._prepare_arrays(buf, out)

        if self.elementsize <= 1:
            return out  # no shuffling needed

        _doUnshuffle(buf.view("uint8"), out.view("uint8"), self.elementsize)

        return out

    def __repr__(self) -> str:
        return f'{type(self).__name__}(elementsize={self.elementsize})'
