import numpy as np
from abc import ABC, abstractmethod


class BaseOp(ABC):

    differentiable: bool

    @abstractmethod
    def forward(self, *buffers: np.ndarray, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        pass


class DiffOp(BaseOp):

    differentiable = True

    def maybe_debroadcast_grad(self, grad: np.ndarray, input: np.ndarray) -> np.ndarray:
        input_shape = input.shape
        grad_shape = grad.shape

        if input_shape == grad_shape:
            return grad

        ndims_grad = len(grad_shape)
        ndims_input = len(input_shape)

        if ndims_grad > ndims_input:
            grad = np.sum(
                grad, axis=tuple(range(ndims_grad - ndims_input)), keepdims=False
            )

        grad_shape = grad.shape
        reduction_axes = []

        for i in range(len(grad_shape)):
            if grad_shape[i] > input_shape[i]:
                reduction_axes.append(i)

        if reduction_axes:
            grad = np.sum(grad, axis=tuple(reduction_axes), keepdims=True)

        return grad


class NonDiffOp(BaseOp):

    differentiable = False

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        raise RuntimeError("Called `backward()` of non-differentiable operation")
