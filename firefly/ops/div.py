from firefly.ops.base import DiffOp
from firefly.backend.base import BaseBuffer
from firefly.backend.dispatcher import BackendDispatcher as D


class Div(DiffOp):
    def forward(self, *buffers: BaseBuffer, **kwargs) -> BaseBuffer:
        assert len(buffers) == 2, "Div requires exactly 2 inputs"
        self.buffers = buffers
        return D.div(buffers[0], buffers[1])

    def backward(self, grad: BaseBuffer) -> tuple[BaseBuffer, BaseBuffer]:
        a, b = self.buffers
        grad_a = D.div(grad, b)  # grad * (1/x2)
        grad_b = D.mul(
            grad,
            D.div(D.neg(a), D.mul(b, b)),
        )  # grad * (-x1/x2^2)
        grad_a = self.maybe_debroadcast_grad(grad_a, a)
        grad_b = self.maybe_debroadcast_grad(grad_b, b)
        return (grad_a, grad_b)
