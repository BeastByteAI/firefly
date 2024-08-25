<h1 align="center">
  <br>
 <img src="https://gist.githubusercontent.com/OKUA1/55e2fb9dd55673ec05281e0247de6202/raw/d7d7ffd7f6b51ba3b314717ea133d31d26c943d2/firefly.svg" alt="FireFly" width="250" height = "250">
  <br>
  FireFly
  <br>
</h1>

<h4 align="center">A DIY Deep Learning Framework</h4>

_FireFly_ is an educational project that aims to provide a simple and easy-to-understand implementation of a deep learning framework. While it is not intended to compete with bigger projects like TensorFlow or PyTorch, it is a great way to get a better intuition of how such frameworks work under the hood.

## How does it work? 

You can re-create the framework completely from scratch by following the [tutorials](https://github.com/BeastByteAI/firefly/tree/main/tutorials). The framework is built on top of NumPy and only takes a couple of hours to implement the first version (covered in `1_foundations` tutorial). At this point, it is already possible to train simple MLPs using a high-level API similar to PyTorch.

If you have any questions or need help, feel free to open an issue or reach out to us on [Discord](https://discord.com/invite/YDAbwuWK7V).

## Contributing

There are many ways to contribute to FireFly:
- Help us extend the framework by implementing new features;
- Improve the existing codebase by fixing bugs or refactoring the code;
- Expand the tutorials by adding new topics or improving the existing ones;
- Help with the documentation;
- Spread the word about FireFly by sharing it with your friends or on social media;
- Provide feedback on how we can improve the project or what features you would like to see in the future.

In addition, you can support the project by simply starring ⭐ the repository on GitHub.

Finally, we have several other projects that you might be interested in:

<br>
<a href="https://github.com/iryna-kondr/scikit-llm">
  <picture>
  <source media="(prefers-color-scheme: light)" srcset="https://gist.githubusercontent.com/OKUA1/ce2167df8e441ce34a9fbc8578b86543/raw/f740c391ec37eaf2f80d5b46f1fa2a989dd45932/skll_h_dark.svg" >
  <source media="(prefers-color-scheme: dark)" srcset="https://gist.githubusercontent.com/OKUA1/ce2167df8e441ce34a9fbc8578b86543/raw/f740c391ec37eaf2f80d5b46f1fa2a989dd45932/skllm_h_light.svg">
  <img alt="Logo" src="https://gist.githubusercontent.com/OKUA1/ce2167df8e441ce34a9fbc8578b86543/raw/f740c391ec37eaf2f80d5b46f1fa2a989dd45932/skll_h_dark.svg" height = "65">
</picture>
</a> <br><br>
<a href="https://github.com/OKUA1/agent_dingo">
  <picture>
  <source media="(prefers-color-scheme: light)" srcset="https://gist.githubusercontent.com/OKUA1/ce2167df8e441ce34a9fbc8578b86543/raw/f740c391ec37eaf2f80d5b46f1fa2a989dd45932/dingo_h_dark.svg" >
  <source media="(prefers-color-scheme: dark)" srcset="https://gist.githubusercontent.com/OKUA1/ce2167df8e441ce34a9fbc8578b86543/raw/f740c391ec37eaf2f80d5b46f1fa2a989dd45932/ding_h_light.svg">
  <img alt="Logo" src="https://gist.githubusercontent.com/OKUA1/ce2167df8e441ce34a9fbc8578b86543/raw/f740c391ec37eaf2f80d5b46f1fa2a989dd45932/dingo_h_dark.svg" height = "65">
</picture>
</a> <br><br>
<a href="https://github.com/OKUA1/falcon">
  <picture>
  <source media="(prefers-color-scheme: light)" srcset="https://gist.githubusercontent.com/OKUA1/ce2167df8e441ce34a9fbc8578b86543/raw/f740c391ec37eaf2f80d5b46f1fa2a989dd45932/falcon_h_dark.svg" >
  <source media="(prefers-color-scheme: dark)" srcset="https://gist.githubusercontent.com/OKUA1/ce2167df8e441ce34a9fbc8578b86543/raw/f740c391ec37eaf2f80d5b46f1fa2a989dd45932/falcon_h_light.svg">
  <img alt="Logo" src="https://gist.githubusercontent.com/OKUA1/ce2167df8e441ce34a9fbc8578b86543/raw/f740c391ec37eaf2f80d5b46f1fa2a989dd45932/dingo_h_dark.svg" height = "65">
</picture>
</a>


## FAQ 

Q: Why should I use FireFly instead of PyTorch or TensorFlow? \
A: FireFly is not intended to be a replacement of PyTorch or TensorFlow. It is an educational project, where you have the opportunity to implement the whole framework from scratch. 

Q: Can I use FireFly in production? \
A: At the moment, FireFly is very minimalistic and lacks many features. In the future we plan to make it more feature-rich, but we are not planning to actively optimize it for speed or memory usage.

Q: Can I contribute to FireFly? \
A: Yes! We are always looking for contributors. Please see the section above.

Q: Will I be able to follow the tutorials without any prior knowledge? \
A: The tutorials are designed to be beginner-friendly, but some basic knowledge is beneficial as we are not covering the theory too much. Please see the preface of `1_foundations` tutorial for more information. In addition, FireFly API is loosely similar to PyTorch, so if you have experience with PyTorch, you will find it easier to follow the tutorials.

Q: Are there any docs available? \
A: At the moment there are no docs as you are expected to implement the framework yourself. However, we are planning to add docs in the future once the framework is more feature-rich.

Q: Can you provide an example of how to use FireFly? \
A: Sure! Here is an example of how to train a simple MLP model using FireFly.
```python
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

X, y = Tensor(...), Tensor(...)

def loss_fn(y_pred, y_true):
    ...

optimizer = SGD(model.parameters(), lr=0.01)

epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.to_numpy()}")
```

## Citation

You can cite FireFly using the following BibTeX:

```
@software{FireFly,
  author = {Oleh Kostromin and Iryna Kondrashchenko},
  year = {2024},
  publisher = {beastbyte.ai},
  address = {Linz, Austria},
  title = {FireFly: A DIY Deep Learning Framework},
  url = {https://github.com/BeastByteAI/firefly }
}
```