# blosc-project/blosc-project/blosc/blosc.pyi

from typing import Optional, Union

class Blosc:
    NOSHUFFLE: int
    SHUFFLE: int
    BITSHUFFLE: int
    AUTOSHUFFLE: int
    max_buffer_size: int

    def __init__(self, cname: str = 'lz4', clevel: int = 5, shuffle: int = SHUFFLE, blocksize: int = 0) -> None:
        ...

    def encode(self, buf: Union[bytes, bytearray]) -> bytes:
        ...

    def decode(self, buf: Union[bytes, bytearray], out: Optional[Union[bytes, bytearray]] = None) -> bytes:
        ...

    def decode_partial(self, buf: Union[bytes, bytearray], start: int, nitems: int, out: Optional[Union[bytes, bytearray]] = None) -> bytes:
        ...

def compress(source: Union[bytes, bytearray], cname: bytes, clevel: int, shuffle: int = Blosc.SHUFFLE, blocksize: int = 0) -> bytes:
    ...

def decompress(source: Union[bytes, bytearray], dest: Optional[Union[bytes, bytearray]] = None) -> bytes:
    ...

def list_compressors() -> list[str]:
    ...

def get_nthreads() -> int:
    ...

def set_nthreads(nthreads: int) -> int:
    ...

def cbuffer_sizes(source: Union[bytes, bytearray]) -> tuple[int, int, int]:
    ...

def cbuffer_complib(source: Union[bytes, bytearray]) -> str:
    ...

def cbuffer_metainfo(source: Union[bytes, bytearray]) -> tuple[int, int, bool]:
    ...