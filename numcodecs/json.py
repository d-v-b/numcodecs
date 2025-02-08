import json as _json
import textwrap
from typing import Any, ClassVar, Literal

import numpy as np
from typing_extensions import Buffer

from .abc import Codec, ConfigDict
from .compat import ensure_text


class JSONConfig(ConfigDict[Literal['json2']]):
    encoding: str
    skipkeys: bool
    ensure_ascii: bool
    check_circular: bool
    allow_nan: bool
    sort_keys: bool
    indent: int | None
    separators: tuple[str, str] | None
    strict: bool


class JSON(Codec[Literal['json2']]):
    """Codec to encode data as JSON. Useful for encoding an array of Python objects.

    .. versionchanged:: 0.6
        The encoding format has been changed to include the array shape in the encoded
        data, which ensures that all object arrays can be correctly encoded and decoded.

    Examples
    --------
    >>> import numcodecs
    >>> import numpy as np
    >>> x = np.array(['foo', 'bar', 'baz'], dtype='object')
    >>> codec = numcodecs.JSON()
    >>> codec.decode(codec.encode(x))
    array(['foo', 'bar', 'baz'], dtype=object)

    See Also
    --------
    numcodecs.pickles.Pickle, numcodecs.msgpacks.MsgPack

    """
    _text_encoding: str
    skipkeys: bool
    strict: bool
    ensure_ascii: bool
    check_circular: bool
    allow_nan: bool
    sort_keys: bool
    separators: tuple[str, str] | None
    _encoder: _json.JSONEncoder

    def __init__(
        self,
        encoding: str = 'utf-8',
        skipkeys: bool = False,
        ensure_ascii: bool = True,
        check_circular: bool = True,
        allow_nan: bool = True,
        sort_keys: bool = True,
        indent: int | None = None,
        separators: tuple[str, str] | None = None,
        strict: bool = True,
    ):
        self._text_encoding = encoding
        if separators is None:
            # ensure separators are explicitly specified, and consistent behaviour across
            # Python versions, and most compact representation if indent is None
            if indent is None:
                separators = ',', ':'
            else:
                separators = ', ', ': '

        self.skipkeys = skipkeys
        self.ensure_ascii = ensure_ascii
        self.check_circular = check_circular
        self.allow_nan = allow_nan
        self.indent = indent
        self.separators = separators
        self.strict = strict
        self.sort_keys = sort_keys
        self._encoder = _json.JSONEncoder(
            separators=self.separators,
            skipkeys=self.skipkeys,
            ensure_ascii=self.ensure_ascii,
            check_circular=self.check_circular,
            allow_nan=self.allow_nan,
            sort_keys=self.sort_keys,
            indent=self.indent,
        )
        self._decoder = _json.JSONDecoder(strict=self.strict)

    def encode(self, buf: Buffer) -> bytes:
        try:
            buf = np.asarray(buf)
        except ValueError:  # pragma: no cover
            buf = np.asarray(buf, dtype=object)
        items = np.atleast_1d(buf).tolist()
        items.append(buf.dtype.str)
        items.append(buf.shape)
        return self._encoder.encode(items).encode(self._text_encoding)

    def decode(self, buf: Buffer, out: Buffer | None = None) -> np.ndarray[Any, np.dtype[Any]]:
        items = self._decoder.decode(ensure_text(buf, self._text_encoding))
        dec = np.empty(items[-1], dtype=items[-2])
        if not items[-1]:
            dec[...] = items[0]
        else:
            dec[:] = items[:-2]
        if out is not None:
            np.copyto(out, dec)
            return out
        else:
            return dec

    def get_config(self) -> JSONConfig:
        config: JSONConfig = {
            'id': self.codec_id,
            'encoding': self._text_encoding,
            'skipkeys': self.skipkeys,
            'ensure_ascii': self.ensure_ascii,
            'check_circular': self.check_circular,
            'allow_nan': self.allow_nan,
            'sort_keys': self.sort_keys,
            'indent': self.indent,
            'separators': self.separators,
            'strict': self.strict,
        }
        return config

    def __repr__(self) -> str:
        params = [f'encoding={self._text_encoding!r}']
        for k, v in sorted(self.__dict__.items()):
            params.append(f'{k}={v!r}')
        classname = type(self).__name__
        params_joined = ', '.join(params)
        return textwrap.fill(
            f'{classname}({params_joined})',
            width=80,
            break_long_words=False,
            subsequent_indent='     ',
        )
