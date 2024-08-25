from firefly.nn.optimizers.base import BaseOptimizer
from firefly.nn.parameter import Parameter
from firefly.tensor import Tensor


class SGD(BaseOptimizer):
    def __init__(self, parameters: list[Parameter], lr=0.01):
        super().__init__(parameters)
        self.lr = Tensor(lr)

    def step(self):
        for param in self.parameters:
            if param.tensor.grad is None:
                raise ValueError("Parameter has no gradient.")
            param.update(param.tensor - self.lr * param.tensor.grad)

    def to(self, backend: str):
        self.lr = self.lr.to(backend)
        return self
