from firefly.ops.base import DiffOp
from firefly.backend.base import BaseBuffer
from firefly.backend.dispatcher import BackendDispatcher as D


class ReLU(DiffOp):
    def forward(self, *buffers: BaseBuffer, **kwargs) -> BaseBuffer:
        assert len(buffers) == 1, "ReLU requires exactly 1 input"
        a = buffers[0]
        self.mask = D.gr(a, 0.0)
        return D.mul(a, self.mask)

    def backward(self, grad: BaseBuffer) -> tuple[BaseBuffer]:
        return (D.mul(grad, self.mask),)
