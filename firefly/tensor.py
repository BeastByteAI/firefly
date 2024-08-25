from __future__ import annotations
from firefly.backend.base import BaseBuffer
from firefly.backend.dispatcher import backends as _backends
from firefly.backend.implementation.numpy import NumpyBuffer
from firefly.ops.base import BaseOp
from firefly.ops import (
    MatMul,
    Mul,
    Add,
    Sub,
    Div,
    ReLU,
    Sigmoid,
    ReduceSum,
    Slice,
    Concat,
    Reshape,
    Sqrt,
    Transpose,
    Softmax,
    GatherND,
    Shape,
    ReduceMean,
    Exp,
    Tanh,
)
from typing import Type
import numpy as np


class Tensor:

    parents: tuple[Tensor] | None = None
    creator: BaseOp | None = None

    def __init__(
        self,
        data: np.ndarray | BaseBuffer,
        requires_grad=False,
        store_grad_non_leaf=False,
    ):
        if isinstance(data, float):
            data = np.array([data], dtype=np.float32)
        elif isinstance(data, int):
            data = np.array([data], dtype=np.int64)
        elif isinstance(data, list):
            data = np.array(data)
        self.buffer: BaseBuffer = (
            data if isinstance(data, BaseBuffer) else NumpyBuffer(data)
        )
        self.requires_grad = requires_grad
        self.store_grad_non_leaf = store_grad_non_leaf
        self.grad = None
        self.creator = None
        self.backend: str = self.buffer.backend + "/" + self.buffer.device

    def to(self, backend: str) -> Tensor:
        if "/" in backend:
            backend, device = backend.split("/")
        else:
            backend, device = backend, None
        backend_impl = _backends[backend]
        self.buffer = backend_impl.buffer(self.buffer, device=device)
        self.backend = self.buffer.backend + "/" + self.buffer.device
        return self

    def backward(self, grad_output=None):
        if not self.requires_grad:
            return

        if grad_output is None:
            if self.buffer.size() == 1:
                grad_output = Tensor(1.0).to(self.backend)
            else:
                raise RuntimeError("Grad must be specified for non-scalar tensors.")

        if not self.parents or self.store_grad_non_leaf:
            if self.grad is None:
                self.grad = grad_output
            else:
                self.grad += grad_output

        if self.creator and self.parents:
            grad_inputs = self.creator.backward(
                grad_output.buffer
            )  # only propagate new grad
            for inp, grad in zip(self.parents, grad_inputs):
                inp.backward(Tensor(grad))

    def zero_grad(self, cascade: bool = False):
        self.grad = None
        if cascade and self.parents:
            for parent in self.parents:
                parent.zero_grad()

    def __repr__(self) -> str:
        return f"Tensor({self.buffer.data_repr()}, shape = {self.buffer.shape()}, backend = {self.backend}, requires_grad = {self.requires_grad})"

    def __matmul__(self, other: Tensor) -> Tensor:
        return apply_op(MatMul, self, other)

    def __mul__(self, other: Tensor) -> Tensor:
        return apply_op(Mul, self, other)

    def __add__(self, other: Tensor) -> Tensor:
        return apply_op(Add, self, other)

    def __sub__(self, other: Tensor) -> Tensor:
        return apply_op(Sub, self, other)

    def __truediv__(self, other: Tensor) -> Tensor:
        return apply_op(Div, self, other)

    def relu(self) -> Tensor:
        return apply_op(ReLU, self)

    def sigmoid(self) -> Tensor:
        return apply_op(Sigmoid, self)

    def reduce_sum(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        return apply_op(ReduceSum, self, axis=axis, keepdims=keepdims)

    def reduce_mean(self, axis: int = None, keepdims: bool = False) -> Tensor:
        return apply_op(ReduceMean, self, axis=axis, keepdims=keepdims)

    def shape(self) -> Tensor:
        return apply_op(Shape, self)

    def slice(
        self, starts: Tensor | list | tuple, ends: Tensor | list | tuple
    ) -> Tensor:
        if not isinstance(starts, Tensor):
            starts = Tensor(np.array(starts)).to(self.backend)
        if not isinstance(ends, Tensor):
            ends = Tensor(np.array(ends)).to(self.backend)
        return apply_op(Slice, self, starts, ends)

    def __getitem__(self, index):
        starts = []
        ends = []
        if isinstance(index, int):
            index = (slice(index, index + 1),)
        if isinstance(index, slice):
            index = (index,)
        if isinstance(index, tuple):
            for slice_ in index:
                start = slice_.start or 0
                end = slice_.stop or 9223372036854775807
                if start < 0:
                    raise NotImplementedError(
                        "Negative slicing indices not yet supported."
                    )
                starts.append(start)
                ends.append(end)
        else:
            raise NotImplementedError("Only tuple slicing is supported.")
        return self.slice(starts, ends)

    def concat(self, other: list[Tensor], axis: int) -> Tensor:
        return apply_op(Concat, self, *other, axis=axis)

    def reshape(self, shape: list[int] | tuple[int]) -> Tensor:
        if not isinstance(shape, Tensor):
            shape = Tensor(np.array(shape)).to(self.backend)
        return apply_op(Reshape, self, shape)

    def square(self) -> Tensor:
        return apply_op(Mul, self, self)

    def sqrt(self) -> Tensor:
        return apply_op(Sqrt, self)

    def to_numpy(self):
        return self.buffer.to_numpy()

    def transpose(self, axes: list[int] | tuple[int] | None = None) -> Tensor:
        return apply_op(Transpose, self, axes=axes)

    @property
    def T(self):
        return self.transpose()

    def softmax(self, axis: int = -1) -> Tensor:
        return apply_op(Softmax, self, axis=axis)

    def gather_nd(self, indices: Tensor) -> Tensor:
        return apply_op(GatherND, self, indices)

    def exp(self) -> Tensor:
        return apply_op(Exp, self)

    def tanh(self) -> Tensor:
        return apply_op(Tanh, self)

    def gelu(self):
        ##### GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

        cubed = self.square() * self
        sqrt_2_pi = Tensor(np.sqrt(2 / np.pi)).to(self.backend)
        c1 = Tensor(0.044715).to(self.backend)
        inner = sqrt_2_pi * (self + c1 * cubed)
        tanh_inner = inner.tanh()
        return (
            Tensor(0.5).to(self.backend)
            * self
            * (Tensor(1.0).to(self.backend) + tanh_inner)
        )


def apply_op(op_class: Type[BaseOp], *args: Tensor, **kwargs):
    op = op_class()
    result_buffer = op.forward(*[arg.buffer for arg in args], **kwargs)
    result = Tensor(
        result_buffer,
        requires_grad=(any((arg.requires_grad for arg in args)) and op.differentiable),
    )
    if result.requires_grad:
        result.creator = op
        result.parents = args
    return result
