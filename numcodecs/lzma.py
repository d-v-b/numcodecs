from types import ModuleType
from typing import ClassVar, Literal, Optional

from typing_extensions import Buffer

_lzma: Optional[ModuleType] = None
try:
    import lzma as _lzma
except ImportError:  # pragma: no cover
    try:  # noqa: SIM105
        from backports import lzma as _lzma  # type: ignore[no-redef, import-not-found]
    except ImportError:
        pass


if _lzma:
    from .abc import Codec
    from .compat import ensure_contiguous_ndarray, ndarray_copy

    # noinspection PyShadowingBuiltins
    class LZMA(Codec[Literal['lzma']]):
        """Codec providing compression using lzma via the Python standard
        library.

        Parameters
        ----------
        format : integer, optional
            One of the lzma format codes, e.g., ``lzma.FORMAT_XZ``.
        check : integer, optional
            One of the lzma check codes, e.g., ``lzma.CHECK_NONE``.
        preset : integer, optional
            An integer between 0 and 9 inclusive, specifying the compression
            level.
        filters : list, optional
            A list of dictionaries specifying compression filters. If
            filters are provided, 'preset' must be None.

        """

        format: int
        check: int
        preset: int | None
        filters: list[object] | None

        def __init__(
            self,
            format: int = 1,
            check: int = -1,
            preset: int | None = None,
            filters: list[object] | None = None,
        ):
            self.format = format
            self.check = check
            self.preset = preset
            self.filters = filters

        def encode(self, buf: Buffer) -> Buffer:
            # normalise inputs
            buf = ensure_contiguous_ndarray(buf)

            # do compression
            return _lzma.compress(
                buf,
                format=self.format,
                check=self.check,
                preset=self.preset,
                filters=self.filters,
            )

        def decode(self, buf: Buffer, out: Buffer | None = None) -> Buffer:
            # normalise inputs
            buf = ensure_contiguous_ndarray(buf)
            if out is not None:
                out = ensure_contiguous_ndarray(out)

            # do decompression
            dec = _lzma.decompress(buf, format=self.format, filters=self.filters)

            # handle destination
            return ndarray_copy(dec, out)

        def __repr__(self) -> str:
            return f'{type(self).__name__}(format={self.format!r}, check={self.check!r}, preset={self.preset!r}, filters={self.filters!r})'
