from firefly.ops.base import DiffOp
from firefly.backend.base import BaseBuffer
from firefly.backend.dispatcher import BackendDispatcher as D
import numpy as np


class Slice(DiffOp):
    def forward(self, *buffers: BaseBuffer, **kwargs) -> BaseBuffer:
        assert len(buffers) == 3, "Slice requires exactly 3 input buffer"
        self.buffer, starts, ends = buffers

        input_shape = self.buffer.shape()

        self.full_starts = [0] * len(input_shape)
        self.full_ends = list(input_shape)

        strats_list = starts.to_numpy().tolist()
        ends_list = ends.to_numpy().tolist()

        self.full_starts[: len(strats_list)] = strats_list
        self.full_ends[: len(ends_list)] = ends_list

        return D.slice(self.buffer, starts=self.full_starts, ends=self.full_ends)

    def backward(self, grad: BaseBuffer) -> BaseBuffer:

        input_shape = self.buffer.shape()
        pads = []

        for length, slice_start, slice_end in zip(
            input_shape, self.full_starts, self.full_ends
        ):
            pads.append(slice_start)
            pads.append(max(0, length - slice_end))
        return (D.pad(grad, pads),)
