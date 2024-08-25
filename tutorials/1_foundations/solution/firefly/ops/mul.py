from firefly.ops.base import DiffOp
import numpy as np


class Mul(DiffOp):
    def forward(self, *buffers: np.ndarray, **kwargs) -> np.ndarray:
        assert len(buffers) == 2, "Mul requires exactly 2 inputs"
        self.inputs = buffers
        return buffers[0] * buffers[1]

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.inputs
        grad_a = self.maybe_debroadcast_grad(grad * b, a)
        grad_b = self.maybe_debroadcast_grad(grad * a, b)
        return (grad_a, grad_b)
