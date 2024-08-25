from firefly.ops.base import DiffOp
from firefly.backend.base import BaseBuffer
from firefly.backend.dispatcher import BackendDispatcher as D
from firefly.backend.implementation.numpy import NumpyBuffer
import numpy as np

LOG2_E = float(np.log2(np.e))

LOG2 = float(np.log(2))


class Softmax(DiffOp):
    def forward(self, *buffers: BaseBuffer, **kwargs) -> BaseBuffer:
        assert len(buffers) == 1, "Softmax requires exactly 1 input"
        self.inputs = buffers[0]

        axis = kwargs.get("axis", -1)
        if axis is not None:
            axis = [axis] if isinstance(axis, int) else axis

        # Shift the input to prevent overflow
        max_val = D.reduce_max(self.inputs, axis=axis, keepdims=True)
        shifted_inputs = D.sub(self.inputs, max_val)

        exp_shifted = D.exp2(D.mul(shifted_inputs, LOG2_E))
        sum_exp_shifted = D.reduce_sum(exp_shifted, axis=axis, keepdims=True)

        result = D.div(exp_shifted, sum_exp_shifted)
        self.forward_result = result
        return result

    def backward(self, grad: BaseBuffer, **kwargs) -> tuple[BaseBuffer]:
        axis = kwargs.get("axis", -1)
        if axis is not None:
            axis = [axis] if isinstance(axis, int) else axis

        softmax_x = self.forward_result

        grad_input = D.mul(
            softmax_x,
            D.sub(
                grad,
                D.reduce_sum(
                    D.mul(grad, softmax_x),
                    axis=axis,
                    keepdims=True,
                ),
            ),
        )
        return (grad_input,)