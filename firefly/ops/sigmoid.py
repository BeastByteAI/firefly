from firefly.ops.base import DiffOp
from firefly.backend.base import BaseBuffer
from firefly.backend.dispatcher import BackendDispatcher as D
import numpy as np

LOG2_E = float(np.log2(np.e))


class Sigmoid(DiffOp):
    def forward(self, *buffers: BaseBuffer, **kwargs) -> BaseBuffer:
        assert len(buffers) == 1, "Sigmoid requires exactly 1 input"
        self.inputs = buffers
        neg_x = D.neg(self.inputs[0])
        neg_x = D.mul(neg_x, LOG2_E)
        exp_neg_x = D.exp2(neg_x)
        one_plus_exp_neg_x = D.add(1.0, exp_neg_x)
        result = D.div(1.0, one_plus_exp_neg_x)
        self.forward_result = result
        return result

    def backward(self, grad: BaseBuffer) -> tuple[BaseBuffer]:
        sigmoid_x = self.forward_result
        one_minus_sigmoid_x = D.sub(1.0, sigmoid_x)
        grad_a = D.mul(grad, D.mul(sigmoid_x, one_minus_sigmoid_x))
        return (grad_a,)
