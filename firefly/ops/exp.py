from firefly.ops.base import DiffOp
from firefly.backend.base import BaseBuffer
from firefly.backend.dispatcher import BackendDispatcher as D
import math


class Exp(DiffOp):
    def forward(self, *buffers: BaseBuffer, **kwargs) -> BaseBuffer:
        assert len(buffers) == 1, "Exp requires exactly 1 input"
        self.buffers = buffers
        ln2_inv = 1 / math.log(2)
        return D.exp2(D.mul(buffers[0], ln2_inv))

    def backward(self, grad: BaseBuffer) -> tuple[BaseBuffer]:
        (x,) = self.buffers
        exp_x = self.forward(x)
        grad_x = D.mul(grad, exp_x)
        grad_x = self.maybe_debroadcast_grad(grad_x, x)
        return (grad_x,)
