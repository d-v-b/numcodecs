import gzip as _gzip
import io
from typing import ClassVar, Literal

from typing_extensions import Buffer

from .abc import Codec, ConfigDict
from .compat import ensure_bytes, ensure_contiguous_ndarray


class GZipConfig(ConfigDict[Literal['gzip']]):
    level: int


class GZip(Codec[Literal['gzip']]):
    """Codec providing gzip compression using zlib via the Python standard library.

    Parameters
    ----------
    level : int
        Compression level.

    """

    level: int

    def __init__(self, level: int = 1) -> None:
        self.level = level

    def encode(self, buf: Buffer) -> bytes:
        # normalise inputs
        buf = ensure_contiguous_ndarray(buf)

        # do compression
        compressed = io.BytesIO()
        with _gzip.GzipFile(fileobj=compressed, mode='wb', compresslevel=self.level) as compressor:
            compressor.write(buf)
        return compressed.getvalue()

    # noinspection PyMethodMayBeStatic
    def decode(self, buf: Buffer, out: Buffer | None = None) -> Buffer:
        # normalise inputs
        # BytesIO only copies if the data is not of `bytes` type.
        # This allows `bytes` objects to pass through without copying.
        buf_bytes = io.BytesIO(ensure_bytes(buf))

        # do decompression
        with _gzip.GzipFile(fileobj=buf_bytes, mode='rb') as decompressor:
            if out is not None:
                out_view = ensure_contiguous_ndarray(out)
                decompressor.readinto(out_view)
                if decompressor.read(1) != b'':
                    raise ValueError("Unable to fit data into `out`")
            else:
                out = decompressor.read()

        return out
