from firefly.ops.base import DiffOp
from firefly.backend.base import BaseBuffer
from firefly.backend.dispatcher import BackendDispatcher as D


class Transpose(DiffOp):
    def forward(
        self, buffer: BaseBuffer, axes: tuple[int, ...] = None, **kwargs
    ) -> BaseBuffer:
        self.buffer = buffer
        self.axes = axes
        return D.transpose(buffer, axes=axes)

    def backward(self, grad: BaseBuffer) -> BaseBuffer:
        if self.axes is None:
            axes = None
        else:
            axes = tuple(self.axes.index(i) for i in range(len(self.axes)))
        return (D.transpose(grad, axes=axes),)
