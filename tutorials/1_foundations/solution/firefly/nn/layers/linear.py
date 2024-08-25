from firefly.tensor import Tensor
from firefly.nn.parameter import Parameter
from firefly.nn.module import Module


class Linear(Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super().__init__()
        self.weight = Parameter(shape=(in_features, out_features))
        if use_bias:
            self.bias = Parameter(shape=(out_features,))

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.weight.tensor
        if hasattr(self, "bias"):
            y = y + self.bias.tensor
        return y
