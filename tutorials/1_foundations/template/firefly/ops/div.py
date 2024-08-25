from firefly.ops.base import DiffOp
import numpy as np


class Div(DiffOp):
    def forward(self, *buffers: np.ndarray, **kwargs) -> np.ndarray:
        assert len(buffers) == 2, "Div requires exactly 2 inputs"
        self.buffers = buffers
        return buffers[0] / buffers[1]

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.buffers
        grad_a = grad / b  # grad * (1/x2)
        grad_b = grad * (-a / (b**2))  # grad * (-x1/x2^2)
        grad_a = self.maybe_debroadcast_grad(grad_a, a)
        grad_b = self.maybe_debroadcast_grad(grad_b, b)
        return (grad_a, grad_b)
