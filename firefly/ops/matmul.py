from firefly.ops.base import DiffOp
from firefly.backend.base import BaseBuffer
from firefly.backend.dispatcher import BackendDispatcher as D


class MatMul(DiffOp):
    def forward(self, *buffers: BaseBuffer, **kwargs) -> BaseBuffer:
        assert len(buffers) == 2, "MatMul requires exactly 2 inputs"
        self.inputs = buffers
        result = D.matmul(self.inputs[0], self.inputs[1])
        return result

    def backward(self, grad: BaseBuffer) -> tuple[BaseBuffer, BaseBuffer]:
        a, b = self.inputs
        axes_a = [i for i in range(len(a.shape()))]
        axes_a[-2], axes_a[-1] = axes_a[-1], axes_a[-2]
        axes_b = [i for i in range(len(b.shape()))]
        axes_b[-2], axes_b[-1] = axes_b[-1], axes_b[-2]

        a_transposed = D.transpose(a, axes=axes_a)
        b_transposed = D.transpose(b, axes=axes_b)

        grad_a = D.matmul(grad, b_transposed)
        grad_b = D.matmul(a_transposed, grad)

        return (
            self.maybe_debroadcast_grad(grad_a, a),
            self.maybe_debroadcast_grad(grad_b, b),
        )
