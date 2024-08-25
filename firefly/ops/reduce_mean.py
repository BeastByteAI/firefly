from firefly.ops.base import DiffOp
from firefly.backend.base import BaseBuffer
from firefly.backend.dispatcher import BackendDispatcher as D


class ReduceMean(DiffOp):
    def forward(
        self, *buffers: BaseBuffer, axis=None, keepdims=False, **kwargs
    ) -> BaseBuffer:
        assert len(buffers) == 1, "ReduceMean requires exactly 1 input"
        self.buffer = buffers[0]
        self.axis = axis
        self.keepdims = keepdims
        return D.reduce_mean(buffers[0], axis=axis, keepdims=keepdims)

    def backward(self, grad: BaseBuffer) -> BaseBuffer:
        raise NotImplementedError("ReduceMean backward not implemented")
