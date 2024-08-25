from __future__ import annotations

try:
    import cupy as cp
except ImportError:
    cp = None
from numpy.typing import ArrayLike
from firefly.backend.base import BaseBuffer
from firefly.backend.implementation._array_api import ArrayApiBackend

is_cupy_available = cp is not None


class CupyBuffer(BaseBuffer):

    default_device = "0"  # for now we do not support multi-gpu
    backend = "cupy"
    native_type = cp.ndarray if cp else None

    def from_numpy(self, data: ArrayLike) -> ArrayLike:
        if cp:
            return cp.asarray(data)
        raise ImportError("CuPy is not installed")

    def to_numpy(self) -> ArrayLike:
        if cp:
            return cp.asnumpy(self.data)
        raise ImportError("CuPy is not installed")

    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def size(self) -> int:
        return self.data.size

    def data_repr(self):
        return str(self.data).replace("\n", "\n" + " " * 7)

    def __repr__(self):
        data_str = str(self.data).replace("\n", "\n")
        return f"CupyBuffer({data_str})"


class CupyBackend(ArrayApiBackend):
    np = cp
    buffer = CupyBuffer

    @classmethod
    def astuple(cls, data: ArrayLike):
        return tuple(cp.asnumpy(data))
