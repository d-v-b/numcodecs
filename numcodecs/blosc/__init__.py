"""
This file initializes the `blosc` package.
"""

from .blosc import Blosc, compress, decompress, list_compressors, get_nthreads, set_nthreads

__all__ = [
    "Blosc",
    "compress",
    "decompress",
    "list_compressors",
    "get_nthreads",
    "set_nthreads",
]