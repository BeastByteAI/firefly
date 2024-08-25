import numpy as np
from firefly.ops.base import DiffOp
from firefly.backend.base import BaseBuffer
from firefly.backend.implementation.numpy import NumpyBuffer
from firefly.backend.dispatcher import BackendDispatcher as D


class GatherND(DiffOp):
    def forward(self, *buffers: BaseBuffer, **kwargs) -> BaseBuffer:
        assert len(buffers) == 2, "GatherND requires exactly 2 inputs"
        data, indices = buffers
        self.data = data
        self.indices = indices
        return D.gather_nd(data, indices)

    def backward(self, grad: BaseBuffer) -> tuple[BaseBuffer, BaseBuffer]:
        grad_data = D.constant_of_shape(D.shape(self.data), 0.0)
        grad_data = D.scatter_nd(grad_data, self.indices, grad)
        grad_indices = D.constant_of_shape(D.shape(self.indices), 0.0)
        return (grad_data, grad_indices)
