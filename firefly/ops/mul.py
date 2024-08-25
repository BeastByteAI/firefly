from firefly.ops.base import DiffOp
from firefly.backend.base import BaseBuffer
from firefly.backend.dispatcher import BackendDispatcher as D


class Mul(DiffOp):
    def forward(self, *buffers: BaseBuffer, **kwargs) -> BaseBuffer:
        assert len(buffers) == 2, "Mul requires exactly 2 inputs"
        self.inputs = buffers
        return D.mul(buffers[0], buffers[1])

    def backward(self, grad: BaseBuffer) -> tuple[BaseBuffer, BaseBuffer]:
        a, b = self.inputs
        grad_a = self.maybe_debroadcast_grad(D.mul(grad, b), a)
        grad_b = self.maybe_debroadcast_grad(D.mul(grad, a), b)
        return (grad_a, grad_b)
