from abc import abstractmethod, ABC
from firefly.backend.base import BaseBuffer
from firefly.backend.dispatcher import BackendDispatcher as D


class BaseOp(ABC):

    differentiable: bool

    @abstractmethod
    def forward(self, *buffers: BaseBuffer, **kwargs) -> BaseBuffer:
        pass

    @abstractmethod
    def backward(self, grad: BaseBuffer) -> tuple[BaseBuffer, ...]:
        pass


class DiffOp(BaseOp):

    differentiable = True

    def maybe_debroadcast_grad(self, grad: BaseBuffer, input: BaseBuffer) -> BaseBuffer:
        input_shape = input.shape()
        grad_shape = grad.shape()
        if input_shape == grad_shape:
            return grad

        ndims_grad = len(grad_shape)
        ndims_input = len(input_shape)

        # Remove extra leading dimensions
        if ndims_grad > ndims_input:
            grad = D.reduce_sum(
                grad, axis=tuple(range(ndims_grad - ndims_input)), keepdims=False
            )
        grad_shape = grad.shape()  # recalculate shape after first reduction
        reduction_axes = []
        for i in range(len(grad_shape)):
            if grad_shape[i] > input_shape[i]:
                reduction_axes.append(i)

        if reduction_axes:
            grad = D.reduce_sum(grad, axis=tuple(reduction_axes), keepdims=True)

        return grad


class NonDiffOp(BaseOp):

    differentiable = False

    def backward(self, grad: BaseBuffer) -> tuple[BaseBuffer, ...]:
        raise RuntimeError("Called `backward()` of non-differentiable operation")
