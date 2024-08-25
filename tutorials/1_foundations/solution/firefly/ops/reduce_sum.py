import numpy as np
from firefly.ops.base import DiffOp


class ReduceSum(DiffOp):
    def forward(
        self, *buffers: np.ndarray, axis=None, keepdims=False, **kwargs
    ) -> np.ndarray:
        assert len(buffers) == 1, "ReduceSum requires exactly 1 input"
        self.buffer = buffers[0]
        self.axis = axis
        self.keepdims = keepdims
        return np.sum(buffers[0], axis=axis, keepdims=keepdims)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        input_shape = self.buffer.shape
        grad_shape = list(input_shape)

        if self.axis is None:
            grad_shape = [1] * len(input_shape)
        else:
            axes = [self.axis] if isinstance(self.axis, int) else self.axis
            for ax in axes:
                grad_shape[ax] = 1

        grad = np.reshape(grad, grad_shape)
        return (np.broadcast_to(grad, input_shape),)
