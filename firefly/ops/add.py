from firefly.ops.base import DiffOp
from firefly.backend.base import BaseBuffer
from firefly.backend.dispatcher import BackendDispatcher as D


class Add(DiffOp):
    def forward(self, *buffers: BaseBuffer, **kwargs) -> BaseBuffer:
        assert len(buffers) == 2, "Add requires exactly 2 inputs"
        self.buffers = buffers
        return D.add(buffers[0], buffers[1])

    def backward(self, grad: BaseBuffer) -> BaseBuffer:
        grad_a = self.maybe_debroadcast_grad(grad, self.buffers[0])
        grad_b = self.maybe_debroadcast_grad(grad, self.buffers[1])
        return (grad_a, grad_b)
