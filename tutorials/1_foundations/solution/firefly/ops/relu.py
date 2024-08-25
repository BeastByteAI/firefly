from firefly.ops.base import DiffOp
import numpy as np


class ReLU(DiffOp):
    def forward(self, *buffers: np.ndarray, **kwargs) -> np.ndarray:
        assert len(buffers) == 1, "ReLU requires exactly 1 input"
        a = buffers[0]
        self.mask = a > 0.0
        return a * self.mask

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray]:
        return (grad * self.mask,)
