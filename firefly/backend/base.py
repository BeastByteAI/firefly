from __future__ import annotations
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod


# This is not used for now and needed just for keeping the count of prims
class Primitives(Enum):
    MATMUL = 0
    ADD = 1
    SUB = 2
    MUL = 4
    DIV = 5
    MAX = 6
    MOD = 7
    GR = 8
    EQ = 9
    TRANSPOSE = 10
    LOG2 = 11
    EXP2 = 12
    SIN = 13
    SQRT = 14
    NEG = 15
    REDUCE_SUM = 16
    REDUCE_MAX = 17
    REDUCE_MEAN = 18
    SHAPE = 19
    RESHAPE = 20
    SLICE = 21
    PAD = 22
    CONCAT = 23
    BROADCAST_TO = 24
    AND = 25
    OR = 26
    TANH = 27
    GATHER_ND = 28
    SCATTER_ND = 29
    CONSTANT_OF_SHAPE = 30


class BaseBackend(ABC):

    @staticmethod
    @abstractmethod
    def matmul(a: BaseBuffer, b: BaseBuffer) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def add(a: BaseBuffer, b: BaseBuffer) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def sub(a: BaseBuffer, b: BaseBuffer) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def mul(a: BaseBuffer, b: BaseBuffer) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def div(a: BaseBuffer, b: BaseBuffer) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def max(a: BaseBuffer, b: BaseBuffer) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def mod(a: BaseBuffer, b: BaseBuffer) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def gr(a: BaseBuffer, b: BaseBuffer) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def eq(a: BaseBuffer, b: BaseBuffer) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def sqrt(a: BaseBuffer) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def transpose(a: BaseBuffer, axes: list[int] | tuple[int] | None) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def sin(a: BaseBuffer) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def log2(a: BaseBuffer) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def exp2(a: BaseBuffer) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def neg(a: BaseBuffer) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def reduce_sum(a: BaseBuffer, axis: BaseBuffer, keepdims: bool) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def reduce_max(a: BaseBuffer, axis: BaseBuffer, keepdims: bool) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def shape(a: BaseBuffer) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def reshape(a: BaseBuffer, shape: BaseBuffer) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def slice(a: BaseBuffer, starts: BaseBuffer, ends: BaseBuffer) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def pad(a: BaseBuffer, pad_width: BaseBuffer) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def concat(*buffers: BaseBuffer, axis: int) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def broadcast_to(a: BaseBuffer, shape: BaseBuffer) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def and_(a: BaseBuffer, b: BaseBuffer) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def or_(a: BaseBuffer, b: BaseBuffer) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def tanh(a: BaseBuffer) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def gather_nd(a: BaseBuffer, indices: BaseBuffer) -> BaseBuffer:
        pass

    @staticmethod
    @abstractmethod
    def scatter_nd(target: BaseBuffer, indices: BaseBuffer, updates: BaseBuffer):
        pass

    @staticmethod
    @abstractmethod
    def constant_of_shape(shape: BaseBuffer, value: int | float):
        pass


class BaseBuffer(ABC):

    backend: BaseBackend
    default_device: str
    native_type: any

    def __init__(self, data: any, device: str = None):
        self.device = device or self.default_device
        if isinstance(data, np.ndarray):
            self.data = self.from_numpy(data)
        elif self.is_same_device(data):
            self.data = data.data
        elif isinstance(data, BaseBuffer):
            self.data = self.from_numpy(data.to_numpy())
        elif isinstance(data, self.native_type):
            self.data = data  # TODO device checking
        else:
            raise ValueError("Unsupported data format encountered")

    def is_same_device(self, other: BaseBuffer) -> bool:
        return isinstance(other, type(self)) and other.device == self.device

    @abstractmethod
    def from_numpy(data: np.ndarray) -> any:
        pass

    @abstractmethod
    def to_numpy() -> np.ndarray:
        pass

    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass

    @abstractmethod
    def size(self) -> int:
        pass
