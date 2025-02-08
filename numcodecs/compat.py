import array
import codecs
from typing import Any, Literal

import numpy as np
from typing_extensions import Buffer

from .ndarray_like import NDArrayLike, is_ndarray_like


def ensure_ndarray_like(buf: Buffer | NDArrayLike) -> NDArrayLike:
    """Convenience function to coerce `buf` to ndarray-like array.

    Parameters
    ----------
    buf : ndarray-like, array-like, or bytes-like
        A numpy array like object such as numpy.ndarray, cupy.ndarray, or
        any object exporting a buffer interface.

    Returns
    -------
    arr : NDArrayLike
        A ndarray-like, sharing memory with `buf`.

    Notes
    -----
    This function will not create a copy under any circumstances, it is guaranteed to
    return a view on memory exported by `buf`.
    """

    if not is_ndarray_like(buf):
        if isinstance(buf, array.array) and buf.typecode in "cu":
            # Guard condition, do not support array.array with unicode type, this is
            # problematic because numpy does not support it on all platforms. Also do not
            # support char as it was removed in Python 3.
            raise TypeError("array.array with char or unicode type is not supported")
        else:
            # N.B., first take a memoryview to make sure that we subsequently create a
            # numpy array from a memory buffer with no copy
            mem = memoryview(buf)
            # instantiate array from memoryview, ensures no copy
            buf = np.array(mem, copy=False)
    return buf


def ensure_ndarray(buf: Buffer | NDArrayLike) -> np.ndarray[Any, np.dtype[Any]]:
    """Convenience function to coerce `buf` to a numpy array, if it is not already a
    numpy array.

    Parameters
    ----------
    buf : array-like or bytes-like
        A numpy array or any object exporting a buffer interface.

    Returns
    -------
    arr : ndarray
        A numpy array, sharing memory with `buf`.

    Notes
    -----
    This function will not create a copy under any circumstances, it is guaranteed to
    return a view on memory exported by `buf`.
    """
    return np.array(ensure_ndarray_like(buf), copy=False)


def ensure_contiguous_ndarray_like(
    buf: Buffer, max_buffer_size: int | None = None, flatten: bool = True
) -> NDArrayLike:
    """Convenience function to coerce `buf` to ndarray-like array.
    Also ensures that the returned value exports fully contiguous memory,
    and supports the new-style buffer interface. If the optional max_buffer_size is
    provided, raise a ValueError if the number of bytes consumed by the returned
    array exceeds this value.

    Parameters
    ----------
    buf : ndarray-like, array-like, or bytes-like
        A numpy array like object such as numpy.ndarray, cupy.ndarray, or
        any object exporting a buffer interface.
    max_buffer_size : int
        If specified, the largest allowable value of arr.nbytes, where arr
        is the returned array.
    flatten : bool
        If True, the array are flatten.

    Returns
    -------
    arr : NDArrayLike
        A ndarray-like, sharing memory with `buf`.

    Notes
    -----
    This function will not create a copy under any circumstances, it is guaranteed to
    return a view on memory exported by `buf`.
    """
    arr = ensure_ndarray_like(buf)

    # check for object arrays, these are just memory pointers, actual memory holding
    # item data is scattered elsewhere
    if arr.dtype == object:
        raise TypeError("object arrays are not supported")

    # check for datetime or timedelta ndarray, the buffer interface doesn't support those
    if arr.dtype.kind in "Mm":
        arr = arr.view(np.int64)  # type: ignore[arg-type]

    # check memory is contiguous, if so flatten
    if arr.flags.c_contiguous or arr.flags.f_contiguous:
        if flatten:
            # can flatten without copy
            arr = arr.reshape(-1, order="A")
    else:
        raise ValueError("an array with contiguous memory is required")

    if max_buffer_size is not None and arr.nbytes > max_buffer_size:
        msg = f"Codec does not support buffers of > {max_buffer_size} bytes"
        raise ValueError(msg)

    return arr


def ensure_contiguous_ndarray(
    buf: Buffer, max_buffer_size: int | None = None, flatten: bool = True
) -> np.ndarray[Any, np.dtype[Any]]:
    """Convenience function to coerce `buf` to a numpy array, if it is not already a
    numpy array. Also ensures that the returned value exports fully contiguous memory,
    and supports the new-style buffer interface. If the optional max_buffer_size is
    provided, raise a ValueError if the number of bytes consumed by the returned
    array exceeds this value.

    Parameters
    ----------
    buf : array-like or bytes-like
        A numpy array or any object exporting a buffer interface.
    max_buffer_size : int
        If specified, the largest allowable value of arr.nbytes, where arr
        is the returned array.
    flatten : bool
        If True, the array are flatten.

    Returns
    -------
    arr : ndarray
        A numpy array, sharing memory with `buf`.

    Notes
    -----
    This function will not create a copy under any circumstances, it is guaranteed to
    return a view on memory exported by `buf`.
    """

    return ensure_ndarray(
        ensure_contiguous_ndarray_like(buf, max_buffer_size=max_buffer_size, flatten=flatten)
    )


def ensure_bytes(buf: Buffer | NDArrayLike) -> bytes:
    """Obtain a bytes object from memory exposed by `buf`."""

    if not isinstance(buf, bytes):
        arr = ensure_ndarray_like(buf)

        # check for object arrays, these are just memory pointers,
        # actual memory holding item data is scattered elsewhere
        if arr.dtype == object:
            raise TypeError("object arrays are not supported")

        # create bytes
        buf = arr.tobytes(order="A")

    return buf


def ensure_text(s: Buffer | str, encoding: str = "utf-8") -> str:
    if not isinstance(s, str):
        s = ensure_contiguous_ndarray(s)
        s = codecs.decode(s, encoding)
    return s


def ndarray_copy(src: NDArrayLike, dst: NDArrayLike | None) -> NDArrayLike:
    """Copy the contents of the array from `src` to `dst`."""

    if dst is None:
        # no-op
        return ensure_contiguous_ndarray_like(src)

    # ensure ndarray like
    src_arr = ensure_ndarray_like(src)
    dst_arr = ensure_ndarray_like(dst)
    order: Literal['C', 'F', 'A']
    # flatten source array
    src_flat = src_arr.reshape(-1, order="A")

    # ensure same data type
    if str(dst_arr.dtype) != 'object':
        src_astype = src_flat.view(dst.dtype)
    else:
        src_astype = src_flat
    # reshape source to match destination
    if src_astype.shape != dst_arr.shape:
        if dst_arr.flags.f_contiguous:
            order = "F"
        else:
            order = "C"
        src_reshaped = src_astype.reshape(dst_arr.shape, order=order)
    else:
        src_reshaped = src_astype
    # copy via numpy

    np.copyto(dst_arr, src_reshaped)

    return dst_arr
