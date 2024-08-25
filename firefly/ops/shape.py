from firefly.ops.base import NonDiffOp
from firefly.backend.base import BaseBuffer
from firefly.backend.dispatcher import BackendDispatcher as D


class Shape(NonDiffOp):

    def forward(self, *buffers: BaseBuffer, **kwargs) -> BaseBuffer:

        assert (
            len(buffers) == 1
        ), "Shape requires exactly 1 input buffers: data and new shape"

        inp = buffers[0]

        return D.shape(inp)
