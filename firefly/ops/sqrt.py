from firefly.ops.base import DiffOp
from firefly.backend.base import BaseBuffer
from firefly.backend.dispatcher import BackendDispatcher as D


class Sqrt(DiffOp):
    def forward(self, *buffers: BaseBuffer, **kwargs) -> BaseBuffer:
        assert len(buffers) == 1, "Sqrt requires exactly 1 input"
        self.input = buffers[0]
        return D.sqrt(self.input)

    def backward(self, grad: BaseBuffer) -> tuple[BaseBuffer, BaseBuffer]:
        return (D.mul(grad, D.div(0.5, D.sqrt(self.input))),)
