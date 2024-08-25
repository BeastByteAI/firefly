from firefly.ops.base import DiffOp
from firefly.backend.base import BaseBuffer
from firefly.backend.dispatcher import BackendDispatcher as D


class Tanh(DiffOp):
    def forward(self, *buffers: BaseBuffer, **kwargs) -> BaseBuffer:
        assert len(buffers) == 1, "Tanh requires exactly 1 input"
        self.input = buffers[0]
        self.tanh_inp = D.tanh(self.input)
        return self.tanh_inp

    def backward(self, grad: BaseBuffer) -> tuple[BaseBuffer, BaseBuffer]:
        inp_squared = D.mul(self.tanh_inp, self.tanh_inp)
        return (
            D.mul(
                grad,
                D.sub(1, inp_squared),
            ),
        )
