import math
import numpy as np
from firefly.tensor import Tensor

def kaiming_uniform(tensor_shape, a=0):
    bound = (
        math.sqrt(3.0)
        * math.sqrt(2.0 / (1 + a**2))
        / math.sqrt(np.prod(tensor_shape[1:]))
    )

    return np.random.uniform(-bound, bound, size=tensor_shape).astype(np.float32)


class Parameter:
    def __init__(
        self,
        data: Tensor | None = None,
        shape: tuple[int, ...] | None = None,
        trainable: bool = True,
    ):
        self.trainable = trainable
        if data is None and shape is None:
            raise ValueError("Either data or shape must be provided")
        if data is not None and shape is not None:
            raise ValueError("Only one of data or shape must be provided")

        self.tensor = data or Tensor(kaiming_uniform(shape), requires_grad=trainable)
        self.tensor.parents = None
        self.tensor.creator = None

    def set_state(self, state):
        self.tensor = Tensor(state, requires_grad=self.tensor.requires_grad)

    def get_state(self):
        return self.tensor.to_numpy()

    def update(self, data: Tensor):
        self.tensor = data
        self.tensor.parents = None
        self.tensor.creator = None