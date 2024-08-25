from firefly.ops.base import DiffOp
from firefly.backend.base import BaseBuffer
from firefly.backend.dispatcher import BackendDispatcher as D


class ReduceSum(DiffOp):
    def forward(
        self, *buffers: BaseBuffer, axis=None, keepdims=False, **kwargs
    ) -> BaseBuffer:
        assert len(buffers) == 1, "ReduceSum requires exactly 1 input"
        self.buffer = buffers[0]
        self.axis = axis
        self.keepdims = keepdims
        return D.reduce_sum(buffers[0], axis=axis, keepdims=keepdims)

    def backward(self, grad: BaseBuffer) -> BaseBuffer:
        input_shape = self.buffer.shape()
        grad_shape = list(input_shape)

        if self.axis is None:
            grad_shape = [1] * len(input_shape)
        else:
            axes = [self.axis] if isinstance(self.axis, int) else self.axis
            for ax in axes:
                grad_shape[ax] = 1

        grad = D.reshape(grad, grad_shape)
        return (D.broadcast_to(grad, input_shape),)
