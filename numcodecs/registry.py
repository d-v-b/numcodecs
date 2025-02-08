"""The registry module provides some simple convenience functions to enable
applications to dynamically register and look-up codec classes."""

import logging
from importlib.metadata import EntryPoints, entry_points

from numcodecs.abc import Codec, ConfigDict
from numcodecs.errors import UnknownCodecError

logger = logging.getLogger("numcodecs")
codec_registry: dict[str, type[Codec[str]]] = {}
entries: dict[str, EntryPoints] = {}


def run_entrypoints() -> None:
    entries.clear()
    eps = entry_points()
    entries.update({e.name: e for e in eps.select(group="numcodecs.codecs")})


run_entrypoints()


def get_codec(config: ConfigDict[str]) -> Codec[str]:
    """Obtain a codec for the given configuration.

    Parameters
    ----------
    config : dict-like
        Configuration object.

    Returns
    -------
    codec : Codec

    Examples
    --------

    >>> import numcodecs as codecs
    >>> codec = codecs.get_codec(dict(id='zlib', level=1))
    >>> codec
    Zlib(level=1)

    """
    config_mut: dict[str, object] = dict(config)
    try:
        codec_id = config_mut.pop('id')
    except KeyError as e:
        raise ValueError("codec configuration must contain an 'id' field") from e
    if not isinstance(codec_id, str):
        raise TypeError(f"codec identifier must be a string, not {type(codec_id)}")

    cls = codec_registry.get(codec_id)
    if cls is None and codec_id in entries:
        logger.debug("Auto loading codec '%s' from entrypoint", codec_id)
        cls = entries[codec_id].load()  # type: ignore[attr-defined]
        register_codec(cls, codec_id=codec_id)
    if cls:
        return cls.from_config(config)
    raise UnknownCodecError(f"{codec_id!r}")


def register_codec(cls: type[Codec], codec_id: str | None = None) -> None:
    """Register a codec class.

    Parameters
    ----------
    cls : Codec class

    Notes
    -----
    This function maintains a mapping from codec identifiers to codec
    classes. When a codec class is registered, it will replace any class
    previously registered under the same codec identifier, if present.

    """
    if codec_id is None:
        codec_id = cls.codec_id
    logger.debug("Registering codec '%s'", codec_id)
    codec_registry[codec_id] = cls
