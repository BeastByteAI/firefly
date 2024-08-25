from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from firefly.backend.base import BaseBuffer
from firefly.backend.implementation._array_api import ArrayApiBackend

class NumpyBuffer(BaseBuffer):

    default_device = "cpu"
    backend = "numpy"
    native_type = np.ndarray

    def from_numpy(self, data: ArrayLike) -> ArrayLike:
        return data

    def to_numpy(self) -> ArrayLike:
        return self.data

    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def size(self) -> int:
        return self.data.size

    def data_repr(self):
        return str(self.data).replace("\n", "\n" + " " * 7)

    def __repr__(self):
        data_str = str(self.data).replace("\n", "\n")
        return f"NumpyBuffer({data_str})"


class NumpyBackend(ArrayApiBackend):
    np = np
    buffer = NumpyBuffer
