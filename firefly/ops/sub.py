from firefly.ops.base import DiffOp
from firefly.backend.base import BaseBuffer
from firefly.backend.dispatcher import BackendDispatcher as D


class Sub(DiffOp):
    def forward(self, *buffers: BaseBuffer, **kwargs) -> BaseBuffer:
        assert len(buffers) == 2, "Sub requires exactly 2 inputs"
        self.buffers = buffers
        return D.sub(buffers[0], buffers[1])

    def backward(self, grad: BaseBuffer) -> tuple[BaseBuffer, BaseBuffer]:
        a, b = self.buffers
        return (
            self.maybe_debroadcast_grad(grad, a),
            self.maybe_debroadcast_grad(D.mul(grad, -1.0), b),
        )
