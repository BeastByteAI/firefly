from __future__ import annotations
import numpy as np
from firefly.ops.base import BaseOp
from firefly.ops import Mul, MatMul, Add, Sub, Div, ReLU, ReduceSum
from typing import Type


class Tensor:

    def __init__(
        self,
        data: np.ndarray,
        requires_grad=False,
        store_grad_non_leaf=False,
    ):
        if isinstance(data, float):
            data = np.array([data], dtype=np.float32)
        elif isinstance(data, int):
            data = np.array([data], dtype=np.int64)
        self.buffer = np.asarray(data)
        self.requires_grad = requires_grad
        self.store_grad_non_leaf = store_grad_non_leaf
        self.grad = None
        self.creator: BaseOp | None = None
        self.parents: tuple[Tensor] | None = None

    def __repr__(self) -> str:
        return f"Tensor({self.buffer}, shape = {self.buffer.shape}, requires_grad = {self.requires_grad})"

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

    def reduce_sum(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        return apply_op(ReduceSum, self, axis=axis, keepdims=keepdims)

    def to_numpy(self) -> np.ndarray:
        return self.buffer.copy()

    def backward(self, grad_output=None):
        if not self.requires_grad:
            return

        if grad_output is None:
            if self.buffer.size == 1:
                grad_output = Tensor(1.0)
            else:
                raise RuntimeError("Grad must be specified for non-scalar tensors.")

        # usually we only need the grads for the leaf tensors
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

    def zero_grad(self):
        self.grad = None


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
