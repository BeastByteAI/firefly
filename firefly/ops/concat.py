from firefly.ops.base import DiffOp
from firefly.backend.base import BaseBuffer
from firefly.backend.dispatcher import BackendDispatcher as D
from numpy import cumsum


class Concat(DiffOp):
    def forward(self, *buffers: BaseBuffer, **kwargs) -> BaseBuffer:
        assert len(buffers) >= 2, "Concat requires at least two input buffers"
        self.axis = kwargs.get("axis", 0)

        self.shapes = [buf.shape() for buf in buffers]  # needed for backward
        concatenated_buffer = D.concat(*buffers, axis=self.axis)
        return concatenated_buffer

    def backward(self, grad: BaseBuffer) -> tuple[BaseBuffer, ...]:
        offsets = cumsum([shape[self.axis] for shape in self.shapes[:-1]]).tolist()
        start_offsets = [0] + offsets
        end_offsets = offsets + [9223372036854775807]

        grads = []
        for buffer_idx, shape in enumerate(self.shapes):
            starts = [0] * len(shape)
            starts[self.axis] = start_offsets[buffer_idx]
            ends = [9223372036854775807] * len(shape)
            ends[self.axis] = end_offsets[buffer_idx]
            grads.append(D.slice(grad, starts, ends))
        return tuple(grads)
