from firefly.ops.base import DiffOp
import numpy as np


class Sub(DiffOp):
    def forward(self, *buffers: np.ndarray, **kwargs) -> np.ndarray:
        assert len(buffers) == 2, "Sub requires exactly 2 inputs"
        self.buffers = buffers
        return buffers[0] - buffers[1]

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.buffers
        return (
            self.maybe_debroadcast_grad(grad, a),
            self.maybe_debroadcast_grad(-grad, b),
        )
