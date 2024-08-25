# Optimizers and Training

In the previous steps we implemented both the lower-level and higher-level components needed to train a neural network model. In this step we will put everything together and train a very simple multi-layer perceptron (MLP) model.

To train the model we will implement a PyTorch-style training loop. A single training step will consist of the following actions:

1. Reset the gradients of the model parameters.
2. Forward pass: Compute the predicted output of the model given the input data.
3. Compute the loss between the predicted output and the target output.
4. Backward pass: Compute the gradients of the loss with respect to the model parameters.
5. Update the model parameters using the gradients.

```python
# torch training loop

# 1) Reset the gradients
optimizer.zero_grad()

# 2) Forward pass
outputs = model(inputs)

# 3) Loss computation
loss = loss_fn(outputs, labels)

# 4) Backward pass
loss.backward()

# 5) Update the model parameters
optimizer.step()
```

To achieve this we are still missing the optimizer class which we will have to implement now. Every optimizer can be seen as a function that takes the model parameters and gradients and computes the updated model parameters.

```python 
# firefly/nn/optimizers/base.py

from firefly.nn.parameter import Parameter


class BaseOptimizer:

    def __init__(self, parameters: list[Parameter]):
        self.parameters = parameters

    def zero_grad(self):
        for param in self.parameters:
            param.tensor.zero_grad()

    def step(self):
        raise NotImplementedError("Step method not implemented.")

```

There are many different optimizers available, we are not going to discuss the details of each one of them. For more information about optimizers you can check the [PyTorch documentation](https://pytorch.org/docs/stable/optim.html).

For this tutorial we will implement the Stochastic Gradient Descent (SGD) optimizer without momentum, which is probably the simplest optimizer available. The SGD optimizer updates the model parameters using the following formula: `w_{t+1} = w_t - lr * g_t`, where `w_t` is the model parameter at time `t`, `lr` is the learning rate and `g_t` is the gradient of the loss with respect to the model parameter at time `t`.

```python
# firefly/nn/optimizers/sgd.py

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

```

Now we can define our model, which will consist of a single hidden layer with 32 units and a ReLU activation function. The input will have 10 features and the output will be a single value (regression problem).

```python
from firefly.nn.module import Module
from firefly.tensor import Tensor
from firefly.nn.layers.linear import Linear
from firefly.nn.optimizers.sgd import SGD
import numpy as np


class MLP(Module):
    def __init__(self, input_size = 10, hidden_size = 32, output_size = 1):
        super().__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        y = self.fc1(x)
        y = y.relu()
        y = self.fc2(y)
        return y
```

Finally, we can generate some training data and train the model: 


```python
N_SAMPLES = 1024
n_samples_tensor = Tensor(float(N_SAMPLES))

X = Tensor(np.random.normal(size=(N_SAMPLES, 10)))
true_weights = np.array(
    [[1.5], [-2.0], [1.0], [1.5], [-2.0], [1.0], [3.0], [-2.0], [1.0], [3.0]]
)
y = Tensor(X.to_numpy() @ true_weights + np.random.normal(size=(N_SAMPLES, 1)) * 0.1)

model = MLP()

def loss_fn(y_pred, y_true):
    loss = y_pred - y_true
    loss = loss * loss
    return loss.reduce_sum()/n_samples_tensor

optimizer = SGD(model.get_parameters(), lr=0.01)

epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.to_numpy()}")
```