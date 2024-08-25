from firefly.nn.module import Module
from firefly.tensor import Tensor
from firefly.nn.layers.linear import Linear
from firefly.nn.optimizers.sgd import SGD
import numpy as np


class MLP(Module):
    def __init__(self, input_size=10, hidden_size=32, output_size=1):
        super().__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        y = self.fc1(x)
        y = y.relu()
        y = self.fc2(y)
        return y


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
    loss = y_pred - y_true
    loss = loss * loss
    return loss.reduce_sum() / n_samples_tensor


optimizer = SGD(model.get_parameters(), lr=0.01)

epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.to_numpy()}")
