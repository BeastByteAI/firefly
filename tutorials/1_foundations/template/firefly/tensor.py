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
        # TODO
        raise NotImplementedError("This method needs to be implemented.")

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
        # TODO
        raise NotImplementedError("This method needs to be implemented.")

    def zero_grad(self):
        self.grad = None


def apply_op(op_class: Type[BaseOp], *args: Tensor, **kwargs):
    # TODO
    raise NotImplementedError("This method needs to be implemented.")
