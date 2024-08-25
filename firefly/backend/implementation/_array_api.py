from numpy.typing import ArrayLike
from types import ModuleType
from firefly.backend.base import BaseBackend, BaseBuffer


class ArrayApiBackend(BaseBackend):

    buffer: BaseBuffer
    np: ModuleType

    @classmethod
    def astuple(cls, data: ArrayLike):
        return tuple(data)

    @classmethod
    def _maybe_promote_to_array(cls, a):
        if not isinstance(a, cls.np.ndarray):
            a = cls.np.asarray([a])
        return a

    @classmethod
    def _reduce(cls, a: ArrayLike, axis: ArrayLike | None, keepdims: bool, func):
        if axis:
            axis = cls.astuple(axis.data)
        reduced = func(a, axis=axis, keepdims=keepdims)
        reduced = cls._maybe_promote_to_array(reduced)
        return cls.buffer(reduced)

    @classmethod
    def matmul(cls, a: BaseBuffer, b: BaseBuffer) -> BaseBuffer:
        c = cls.np.matmul(a.data, b.data)
        c = cls._maybe_promote_to_array(c)
        return cls.buffer(c)

    @classmethod
    def add(cls, a: BaseBuffer, b: BaseBuffer) -> BaseBuffer:
        return cls.buffer(cls.np.add(a.data, b.data))

    @classmethod
    def sub(cls, a: BaseBuffer, b: BaseBuffer) -> BaseBuffer:
        return cls.buffer(cls.np.subtract(a.data, b.data))

    @classmethod
    def mul(cls, a: BaseBuffer, b: BaseBuffer) -> BaseBuffer:
        return cls.buffer(cls.np.multiply(a.data, b.data))

    @classmethod
    def div(cls, a: BaseBuffer, b: BaseBuffer) -> BaseBuffer:
        return cls.buffer(cls.np.divide(a.data, b.data))

    @classmethod
    def max(cls, a: BaseBuffer, b: BaseBuffer) -> BaseBuffer:
        return cls.buffer(cls.np.maximum(a.data, b.data))

    @classmethod
    def mod(cls, a: BaseBuffer, b: BaseBuffer) -> BaseBuffer:
        return cls.buffer(cls.np.mod(a.data, b.data))

    @classmethod
    def gr(cls, a: BaseBuffer, b: BaseBuffer) -> BaseBuffer:
        return cls.buffer(cls.np.greater(a.data, b.data))

    @classmethod
    def eq(cls, a: BaseBuffer, b: BaseBuffer) -> BaseBuffer:
        return cls.buffer(cls.np.equal(a.data, b.data))

    @classmethod
    def transpose(
        cls, a: BaseBuffer, axes: list[int] | tuple[int] | None
    ) -> BaseBuffer:
        return cls.buffer(cls.np.transpose(a.data, axes=axes))

    @classmethod
    def log2(cls, a: BaseBuffer) -> BaseBuffer:
        return cls.buffer(cls.np.log2(a.data))

    @classmethod
    def exp2(cls, a: BaseBuffer) -> BaseBuffer:
        return cls.buffer(cls.np.exp2(a.data))

    @classmethod
    def sin(cls, a: BaseBuffer) -> BaseBuffer:
        return cls.buffer(cls.np.sin(a.data))

    @classmethod
    def sqrt(cls, a: BaseBuffer) -> BaseBuffer:
        return cls.buffer(cls.np.sqrt(a.data))

    @classmethod
    def neg(cls, a: BaseBuffer) -> BaseBuffer:
        return cls.buffer(cls.np.negative(a.data))

    @classmethod
    def reduce_sum(
        cls, a: BaseBuffer, axis: BaseBuffer | None, keepdims: bool
    ) -> BaseBuffer:
        return cls._reduce(a.data, axis, keepdims, cls.np.sum)

    @classmethod
    def reduce_mean(
        cls, a: BaseBuffer, axis: BaseBuffer | None, keepdims: bool
    ) -> BaseBuffer:
        return cls._reduce(a.data, axis, keepdims, cls.np.mean)

    @classmethod
    def reduce_max(
        cls, a: BaseBuffer, axis: BaseBuffer | None, keepdims: bool
    ) -> BaseBuffer:
        return cls._reduce(a.data, axis, keepdims, cls.np.max)

    @classmethod
    def shape(cls, a: BaseBuffer) -> BaseBuffer:
        return cls.buffer(cls.np.array(a.data.shape))

    @classmethod
    def reshape(cls, a: BaseBuffer, shape: BaseBuffer) -> BaseBuffer:
        return cls.buffer(cls.np.reshape(a.data, shape.data.tolist()))

    @classmethod
    def slice(cls, a: BaseBuffer, starts: BaseBuffer, ends: BaseBuffer) -> BaseBuffer:
        slices = [slice(starts.data[i], ends.data[i]) for i in range(len(starts.data))]
        return cls.buffer(a.data[cls.astuple(slices)])

    @classmethod
    def pad(cls, a: BaseBuffer, pad_width: BaseBuffer) -> BaseBuffer:
        pads = [
            (int(pad_width.data[i]), int(pad_width.data[i + 1]))
            for i in range(0, len(pad_width.data), 2)
        ]
        return cls.buffer(cls.np.pad(a.data, pads))

    @classmethod
    def concat(cls, *buffers: BaseBuffer, axis: int) -> BaseBuffer:
        return cls.buffer(cls.np.concatenate([b.data for b in buffers], axis=axis))

    @classmethod
    def broadcast_to(cls, a: BaseBuffer, shape: BaseBuffer) -> BaseBuffer:
        return cls.buffer(cls.np.broadcast_to(a.data, shape.data.tolist()))

    @classmethod
    def and_(cls, a: BaseBuffer, b: BaseBuffer) -> BaseBuffer:
        return cls.buffer(cls.np.logical_and(a.data, b.data))

    @classmethod
    def or_(cls, a: BaseBuffer, b: BaseBuffer) -> BaseBuffer:
        return cls.buffer(cls.np.logical_or(a.data, b.data))

    @classmethod
    def tanh(cls, a: BaseBuffer) -> BaseBuffer:
        return cls.buffer(cls.np.tanh(a.data))

    @classmethod
    def gather_nd(cls, a: BaseBuffer, indices: BaseBuffer) -> BaseBuffer:
        return cls.buffer(a.data[tuple(indices.data.T)])

    @classmethod
    def scatter_nd(cls, target: BaseBuffer, indices: BaseBuffer, updates: BaseBuffer):
        # all prims are currently not inplace
        target = target.data.copy()
        target[tuple(indices.data.T)] = updates.data
        return cls.buffer(target)

    @classmethod
    def constant_of_shape(cls, shape: BaseBuffer, value: int | float) -> BaseBuffer:
        dtype = cls.np.float32 if isinstance(value, float) else cls.np.int32
        return cls.buffer(cls.np.full(cls.astuple(shape.data), value).astype(dtype))
