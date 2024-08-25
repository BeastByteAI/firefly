from firefly.backend.base import BaseBuffer
from firefly.backend.implementation.numpy import NumpyBackend
from firefly.backend.implementation.cupy import CupyBackend, is_cupy_available
import numpy as np

backends = {
    "numpy": NumpyBackend,
    "cpu": NumpyBackend,
}

if is_cupy_available:
    backends["cupy"] = CupyBackend
    backends["cuda"] = CupyBackend
    backends["gpu"] = CupyBackend


class BackendDispatcher:

    @staticmethod
    def _check_device(a: BaseBuffer, b: BaseBuffer):
        if not a.is_same_device(b):
            raise RuntimeError("Encountered data on different devices.", a, b)

    @staticmethod
    def _assert_one_buffer(a, b):
        if not isinstance(a, BaseBuffer) and not isinstance(b, BaseBuffer):
            raise ValueError(
                f"Expected at least one buffer, got {type(a)} and {type(b)}"
            )

    @staticmethod
    def _assert_buffer(a):
        if not isinstance(a, BaseBuffer):
            raise ValueError(f"Expected a buffer, got {type(a)}")

    @staticmethod
    def _make_buffer(data: int | float | list | tuple, base: BaseBuffer) -> BaseBuffer:
        if isinstance(data, int):
            data = np.asarray([data])
        elif isinstance(data, float):
            data = np.asarray([data]).astype(np.float32)
        elif isinstance(data, (list, tuple)):
            data = np.asarray(data)
        return base.__class__(data, device=base.device)

    @staticmethod
    def _promote_to_buffers(a, b) -> tuple[BaseBuffer, BaseBuffer]:
        BackendDispatcher._assert_one_buffer(a, b)
        if isinstance(a, (int, float, list, tuple)):
            a = BackendDispatcher._make_buffer(a, b)
        elif isinstance(b, (int, float, list, tuple)):
            b = BackendDispatcher._make_buffer(b, a)
        return a, b

    @staticmethod
    def _prepare_reduce_axis(
        axis: BaseBuffer | list[int] | tuple[int] | None, a: BaseBuffer
    ) -> BaseBuffer | None:
        if axis is not None:
            if isinstance(axis, int):
                axis = [axis]
            if isinstance(axis, (list, tuple)):
                axis = a.__class__(np.asarray(axis), device=a.device)
            BackendDispatcher._assert_buffer(axis)
        return axis

    ############ PRIMS ############

    @staticmethod
    def matmul(a: BaseBuffer, b: BaseBuffer) -> BaseBuffer:
        if not isinstance(a, BaseBuffer) or not isinstance(b, BaseBuffer):
            raise ValueError(f"Expected two buffers, got {type(a)} and {type(b)}")
        BackendDispatcher._check_device(a, b)
        backend = backends[a.backend]
        return backend.matmul(a, b)

    def add(a: BaseBuffer | float | int, b: BaseBuffer | float | int):
        a, b = BackendDispatcher._promote_to_buffers(a, b)
        BackendDispatcher._check_device(a, b)
        backend = backends[a.backend]
        return backend.add(a, b)

    @staticmethod
    def sub(a: BaseBuffer | float | int, b: BaseBuffer | float | int):
        a, b = BackendDispatcher._promote_to_buffers(a, b)
        BackendDispatcher._check_device(a, b)
        backend = backends[a.backend]
        return backend.sub(a, b)

    @staticmethod
    def mul(a: BaseBuffer | float | int, b: BaseBuffer | float | int):
        a, b = BackendDispatcher._promote_to_buffers(a, b)
        BackendDispatcher._check_device(a, b)
        backend = backends[a.backend]
        return backend.mul(a, b)

    @staticmethod
    def div(a: BaseBuffer | float | int, b: BaseBuffer | float | int):
        a, b = BackendDispatcher._promote_to_buffers(a, b)
        BackendDispatcher._check_device(a, b)
        backend = backends[a.backend]
        return backend.div(a, b)

    @staticmethod
    def max(a: BaseBuffer | float | int, b: BaseBuffer | float | int):
        a, b = BackendDispatcher._promote_to_buffers(a, b)
        BackendDispatcher._check_device(a, b)
        backend = backends[a.backend]
        return backend.max(a, b)

    @staticmethod
    def mod(a: BaseBuffer | float | int, b: BaseBuffer | float | int):
        a, b = BackendDispatcher._promote_to_buffers(a, b)
        BackendDispatcher._check_device(a, b)
        backend = backends[a.backend]
        return backend.mod(a, b)

    @staticmethod
    def gr(a: BaseBuffer | float | int, b: BaseBuffer | float | int):
        a, b = BackendDispatcher._promote_to_buffers(a, b)
        BackendDispatcher._check_device(a, b)
        backend = backends[a.backend]
        return backend.gr(a, b)

    @staticmethod
    def eq(a: BaseBuffer | float | int, b: BaseBuffer | float | int):
        a, b = BackendDispatcher._promote_to_buffers(a, b)
        BackendDispatcher._check_device(a, b)
        backend = backends[a.backend]
        return backend.eq(a, b)

    @staticmethod
    def transpose(a: BaseBuffer, axes: list[int] | tuple[int] | None):
        if axes is not None:
            if not isinstance(axes, (list, tuple)):
                raise ValueError(
                    f"Expected an axes to be a list or tuple, got {type(axes)}"
                )
            for ind in axes:
                if not isinstance(ind, int):
                    raise ValueError(f"Invalid index `{ind}`")
        backend = backends[a.backend]
        return backend.transpose(a, axes=axes)

    @staticmethod
    def log2(a: BaseBuffer):
        BackendDispatcher._assert_buffer(a)
        backend = backends[a.backend]
        return backend.log2(a)

    @staticmethod
    def exp2(a: BaseBuffer):
        BackendDispatcher._assert_buffer(a)
        backend = backends[a.backend]
        return backend.exp2(a)

    @staticmethod
    def sin(a: BaseBuffer):
        BackendDispatcher._assert_buffer(a)
        backend = backends[a.backend]
        return backend.sin(a)

    @staticmethod
    def sqrt(a: BaseBuffer):
        BackendDispatcher._assert_buffer(a)
        backend = backends[a.backend]
        return backend.sqrt(a)

    @staticmethod
    def neg(a: BaseBuffer):
        BackendDispatcher._assert_buffer(a)
        backend = backends[a.backend]
        return backend.neg(a)

    @staticmethod
    def reduce_sum(
        a: BaseBuffer, axis: BaseBuffer | list[int] | tuple | None, keepdims: bool
    ):
        axis = BackendDispatcher._prepare_reduce_axis(axis, a)
        backend = backends[a.backend]
        return backend.reduce_sum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def reduce_mean(
        a: BaseBuffer, axis: BaseBuffer | list[int] | tuple | None, keepdims: bool
    ):
        axis = BackendDispatcher._prepare_reduce_axis(axis, a)
        backend = backends[a.backend]
        return backend.reduce_mean(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def reduce_max(
        a: BaseBuffer, axis: BaseBuffer | list[int] | tuple | None, keepdims: bool
    ):
        axis = BackendDispatcher._prepare_reduce_axis(axis, a)
        backend = backends[a.backend]
        return backend.reduce_max(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def shape(a: BaseBuffer):
        BackendDispatcher._assert_buffer(a)
        backend = backends[a.backend]
        return backend.shape(a)

    @staticmethod
    def reshape(
        a: BaseBuffer,
        shape: BaseBuffer | tuple[int] | list[int],
    ):
        if isinstance(shape, (tuple, list)):
            shape = a.__class__(np.asarray(shape), device=a.device)
        else:
            BackendDispatcher._assert_buffer(shape)
            BackendDispatcher._check_device(a, shape)
        backend = backends[a.backend]
        return backend.reshape(a, shape)

    @staticmethod
    def slice(
        a: BaseBuffer,
        starts: BaseBuffer | list | tuple,
        ends: BaseBuffer | list | tuple,
    ):
        BackendDispatcher._assert_buffer(a)
        starts, _ = BackendDispatcher._promote_to_buffers(starts, a)
        ends, _ = BackendDispatcher._promote_to_buffers(ends, a)
        BackendDispatcher._check_device(a, starts)
        BackendDispatcher._check_device(a, ends)
        backend = backends[a.backend]
        return backend.slice(a, starts, ends)

    @staticmethod
    def pad(a: BaseBuffer, pad_width: BaseBuffer | list | tuple):
        BackendDispatcher._assert_buffer(a)
        pad_width, _ = BackendDispatcher._promote_to_buffers(pad_width, a)
        BackendDispatcher._check_device(a, pad_width)
        backend = backends[a.backend]
        return backend.pad(a, pad_width)

    @staticmethod
    def concat(*buffers: BaseBuffer, axis: int):
        for buffer in buffers:
            BackendDispatcher._assert_buffer(buffer)
        backend = backends[buffers[0].backend]
        return backend.concat(*buffers, axis=axis)

    @staticmethod
    def broadcast_to(a: BaseBuffer, shape: BaseBuffer | tuple[int] | list[int]):
        if isinstance(shape, (tuple, list)):
            shape = a.__class__(np.asarray(shape), device=a.device)
        else:
            BackendDispatcher._assert_buffer(shape)
            BackendDispatcher._check_device(a, shape)
        backend = backends[a.backend]
        return backend.broadcast_to(a, shape)

    @staticmethod
    def and_(a: BaseBuffer, b: BaseBuffer):
        BackendDispatcher._assert_buffer(a)
        BackendDispatcher._assert_buffer(b)
        BackendDispatcher._check_device(a, b)
        backend = backends[a.backend]
        return backend.and_(a, b)

    @staticmethod
    def or_(a: BaseBuffer, b: BaseBuffer):
        BackendDispatcher._assert_buffer(a)
        BackendDispatcher._assert_buffer(b)
        BackendDispatcher._check_device(a, b)
        backend = backends[a.backend]
        return backend.or_(a, b)

    @staticmethod
    def tanh(a: BaseBuffer):
        BackendDispatcher._assert_buffer(a)
        backend = backends[a.backend]
        return backend.tanh(a)

    @staticmethod
    def gather_nd(a: BaseBuffer, indices: BaseBuffer):
        backend = backends[a.backend]
        return backend.gather_nd(a, indices)

    @staticmethod
    def scatter_nd(target: BaseBuffer, indices: BaseBuffer, updates: BaseBuffer):
        backend = backends[target.backend]
        return backend.scatter_nd(target, indices, updates)

    @staticmethod
    def constant_of_shape(shape: BaseBuffer, value: int | float):
        backend = backends[shape.backend]
        return backend.constant_of_shape(shape, value)
