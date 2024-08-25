from firefly.ops.base import DiffOp
from firefly.backend.base import BaseBuffer
from firefly.backend.dispatcher import BackendDispatcher as D


class Reshape(DiffOp):
    def forward(self, *buffers: BaseBuffer, **kwargs) -> BaseBuffer:
        assert (
            len(buffers) == 2
        ), "Reshape requires exactly 2 input buffers: data and new shape"
        inp, new_shape = buffers

        self.input = inp

        return D.reshape(inp, new_shape)

    def backward(self, grad: BaseBuffer) -> BaseBuffer:
        return (D.reshape(grad, self.input.shape()),)
