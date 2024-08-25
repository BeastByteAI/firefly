from firefly.ops.base import DiffOp
import numpy as np


class Add(DiffOp):
    def forward(self, *buffers: np.ndarray, **kwargs) -> np.ndarray:
        assert len(buffers) == 2, "Add requires exactly 2 inputs"
        self.buffers = buffers
        return buffers[0] + buffers[1]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad_a = self.maybe_debroadcast_grad(grad, self.buffers[0])
        grad_b = self.maybe_debroadcast_grad(grad, self.buffers[1])
        return (grad_a, grad_b)
