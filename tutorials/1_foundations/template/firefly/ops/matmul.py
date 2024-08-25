from firefly.ops.base import DiffOp
import numpy as np


import numpy as np
from firefly.ops.base import DiffOp


class MatMul(DiffOp):
    def forward(self, *buffers: np.ndarray, **kwargs) -> np.ndarray:
        assert len(buffers) == 2, "MatMul requires exactly 2 inputs"
        self.inputs = buffers
        result = self.inputs[0] @ self.inputs[1]
        return result

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.inputs

        axes_a = list(range(a.ndim))
        axes_a[-2], axes_a[-1] = axes_a[-1], axes_a[-2]
        a_transposed = np.transpose(a, axes=axes_a)

        axes_b = list(range(b.ndim))
        axes_b[-2], axes_b[-1] = axes_b[-1], axes_b[-2]
        b_transposed = np.transpose(b, axes=axes_b)

        grad_a = np.matmul(grad, b_transposed)
        grad_b = np.matmul(a_transposed, grad)

        return (
            self.maybe_debroadcast_grad(grad_a, a),
            self.maybe_debroadcast_grad(grad_b, b),
        )
