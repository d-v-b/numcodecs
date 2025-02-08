import pickle
from typing import ClassVar, Literal

import numpy as np
from typing_extensions import Buffer

from .abc import Codec, ConfigDict
from .compat import ensure_contiguous_ndarray


class PickleConfig(ConfigDict[Literal['pickle']]):
    protocol: int


class Pickle(Codec[Literal['pickle']]):
    """Codec to encode data as as pickled bytes. Useful for encoding an array of Python string
    objects.

    Parameters
    ----------
    protocol : int, defaults to pickle.HIGHEST_PROTOCOL
        The protocol used to pickle data.

    Examples
    --------
    >>> import numcodecs as codecs
    >>> import numpy as np
    >>> x = np.array(['foo', 'bar', 'baz'], dtype='object')
    >>> f = codecs.Pickle()
    >>> f.decode(f.encode(x))
    array(['foo', 'bar', 'baz'], dtype=object)

    See Also
    --------
    numcodecs.msgpacks.MsgPack

    """
    protocol: int

    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
        self.protocol = protocol

    def encode(self, buf: Buffer) -> Buffer:
        return pickle.dumps(buf, protocol=self.protocol)

    def decode(self, buf: Buffer, out: Buffer | None = None) -> object:
        buf = ensure_contiguous_ndarray(buf)
        dec = pickle.loads(buf)

        if out is not None:
            np.copyto(out, dec)
            return out
        else:
            return dec

    def get_config(self) -> PickleConfig:
        return {'id': self.codec_id, 'protocol': self.protocol}

    def __repr__(self) -> str:
        return f'Pickle(protocol={self.protocol})'
