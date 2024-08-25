from firefly.ops.base import DiffOp
import numpy as np


class Mul(DiffOp):
    def forward(self, *buffers: np.ndarray, **kwargs) -> np.ndarray:
        # TODO
        raise NotImplementedError("This method needs to be implemented.")

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # TODO
        raise NotImplementedError("This method needs to be implemented.")