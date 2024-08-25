from firefly.nn.module import Module
from firefly.tensor import Tensor
from firefly.nn.layers.linear import Linear
from firefly.nn.optimizers.sgd import SGD
import numpy as np


class MLP(Module):
    def __init__(self, input_size=10, hidden_size=32, output_size=1):
        # TODO
        raise NotImplementedError("This method needs to be implemented.")

    def forward(self, x: Tensor) -> Tensor:
        # TODO
        raise NotImplementedError("This method needs to be implemented.")


### data gen start
np.random.seed(42)
N_SAMPLES = 1024
n_samples_tensor = Tensor(float(N_SAMPLES))

X = Tensor(np.random.normal(size=(N_SAMPLES, 10)))
true_weights = np.array(
    [[1.5], [-2.0], [1.0], [1.5], [-2.0], [1.0], [3.0], [-2.0], [1.0], [3.0]]
)
y = Tensor(X.to_numpy() @ true_weights + np.random.normal(size=(N_SAMPLES, 1)) * 0.1)

### data gen end

model = MLP()


def loss_fn(y_pred, y_true):
    # TODO
    raise NotImplementedError("Loss function needs to be implemented.")


optimizer = ...  # TODO

epochs = 10
for epoch in range(epochs):
    # TODO
    raise NotImplementedError("This part needs to be implemented.")
