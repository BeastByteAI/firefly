from firefly.tensor import Tensor
from firefly.nn.parameter import Parameter
from firefly.nn.module import Module
import numpy as np


class BatchNorm2D(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False):
        super().__init__()

        self.num_features = num_features
        self.affine = affine

        self.eps = Tensor(eps)
        self.momentum = Tensor(momentum)
        self.one_minus_momentum = Tensor(1 - momentum)

        self.training = True

        self.one = Tensor(1.0)

        self.running_mean = Parameter(
            Tensor(np.zeros(num_features), requires_grad=False), trainable=False
        )
        self.running_var = Parameter(
            Tensor(np.ones(num_features), requires_grad=False), trainable=False
        )

        if self.affine:
            self.weight = Parameter(shape=(num_features,))
            self.bias = Parameter(shape=(num_features,))

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            batch_size = Tensor(x.buffer.shape()[0])

            mean = x.reduce_sum(axis=0) / batch_size
            var = ((x - mean).square()).reduce_sum(axis=0) / batch_size

            self.running_mean.tensor = (
                self.momentum * mean
                + self.one_minus_momentum * self.running_mean.tensor
            )
            self.running_var.tensor = (
                self.momentum * var + self.one_minus_momentum * self.running_var.tensor
            )
        else:
            mean = self.running_mean.tensor
            var = self.running_var.tensor

        std = (var + self.eps).sqrt()
        x_hat = (x - mean) / std

        if self.affine:
            x_hat = self.weight.tensor * x_hat + self.bias.tensor

        return x_hat
