# Modules and Layers

In the previous chapter we implemented a `Tensor` class that can be used for performing both forward and backward computations. However, our framework still does not allow to express complex neural network architectures in a compact way.

Usually, major deep learning frameworks have either additional sub-modules or stand-alone libraries that bring this functionality. For example, `torch` is most commonly used in a combination with `nn`, `tensorflow` is used with `keras` and `jax` is used with `flax`.

Following these examples, we will implement our own neural networks (`nn`) submodule.

Similarly to PyTorch, we are going to have a `Module` class that can be used to represent both layers and models. 

We can start with the implementation of the forward pass logic, which is the core component of every module, but essentially requires only 4 lines of code.

```python
# firefly/nn/module.py

class Module:
    def forward(self, *ars, **kwargs) -> any:
        raise NotImplementedError("Forward method is not implemented.")

    def __call__(self, *args):
        return self.forward(*args)
```

Now we can use the module class to build a simple linear layer:

```python 
from firefly.nn.module import Module

class Linear(Module):
    def forward(self, x, w, b):
        return x@w + b

m = Linear()
x = Tensor([[2., 4., 6.]])
w = Tensor([[2.], [2.], [2.]])
b = Tensor(1.)
print(m(x))
# 25 
```

So far everything looks great, but at the moment our module is stateless and has no mechanism to automatically keep track of the parameters. Before implementing the parameter tracking logic, let's first define the `Parameter` itself. For that, we are going to use a small container class that would store the value of the parameter as `Tensor` and a boolean flag that indicates whether the parameter should be trainable.

```python
# firefly/nn/parameter.py

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
```

In addition to the `Parameter` class itself, we also implement a small utility function that creates an array of a given shape using the Kaiming Uniform initialization strategy.

As a part of this tutorial we are not going to go any deeper into the discussion of the initialization strategies and will always use `kaiming_uniform` (with gain factor appropriate for ReLU activation). To find out more about the initialization, you can check the following resources:
- [DeepLearning.AI : Initializing neural networks](https://www.deeplearning.ai/ai-notes/initialization/index.html)
- [Papers With Code : Kaiming Initialization](https://paperswithcode.com/method/he-initialization)
- [pytorch.org : torch.nn.init](https://pytorch.org/docs/stable/nn.init.html) 


The `Parameter` class only has 3 methods:
- `set_state` - returns the state of the `Parameter` as a numpy array;
- `get_state` - receives the state of the parameter as a numpy array, instantiates a new `Tensor` and stores it;
- `update` - receives the updated value as a `Tensor`, stores it internally and makes the tensor a leaf node in the computational graph by removing the references to the parents and the creator operator. Note: usually the parameter is updated by changing the value of the tensor inplace. However, we do not have inplace operators in FireFly for the sake of simplicity.

Once we have the implementation of the `Parameter` container we can revisit our `Linear` layer. 

```python
# firefly/nn/layers/linear.py

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
```

Now that we have a dedicated class representing parameters, it is time to proceed with the parameters tracking. 

**First things first, what is parameter tracking and why do we need it?**

As our goal is to iteratively optimize the parameters by updating their values, we need to obtain the list of all trainable parameters in our module. One might say that this list can easily be constructed manually. While true for simple modules like a `Linear` layer we have above, it might not be the case for more complex modules. After all, an arbitrary models might not have parameters assigned to attributes directly, but instead: 
- lists of parameters;
- sub-modules (and sub-sub-modules, and sub-sub-sub modules ...);
- lists of sub-modules.

Therefore, we need a mechanism for automatically finding the parameters regardless of how deeply they are nested within the sub-modules. In order to do that, we can add two additional dictionaries to our `Module`, that will store parameters and sub-modules respectively.


```python 
# firefly/nn/module.py

from __future__ import annotations
from firefly.nn.parameter import Parameter

class Module:

    _parameters: (
        dict[str, Parameter | list[Parameter], tuple[Parameter, ...]] | None
    ) = None
    _modules: dict[str, Module | list[Module] | tuple[Module, ...]] | None = None

    def _setup(self):
        if self._parameters is None:
            object.__setattr__(self, "_parameters", {})
        if self._modules is None:
            object.__setattr__(self, "_modules", {})

    ...
```


Next, we need these dictionaries to be populated automatically if either parameters or modules (or their lists) are assigned as instance attributes. We can achieve that by overriding the `__setattr__` method:

```python 
# firefly/nn/module.py

class Tensor:

    ...

    def __setattr__(self, key, value):
        self._setup()
        # Make sure `_parameteres` and `_modules` are not re-assigned
        if key in {"_parameters", "_modules"}:
            raise ValueError(f"Cannot set attribute {key}.")
        # if the key is already stored remove it for now (e.g. if a Parameter is replaced with e.g. int)
        if key in self._parameters:
            del self._parameters[key]
        if key in self._modules:
            del self._modules[key]
        # Check if the value is a sequence of Parameters/Modules. Reject mixed sequences.
        if isinstance(value, (list, tuple)):
            n_params = len([i for i in value if isinstance(i, Parameter)])
            n_modules = len([i for i in value if isinstance(i, Module)])
            if n_params == len(value):
                self._parameters[key] = value
            elif n_modules == len(value):
                self._modules[key] = value
            elif n_modules > 0 or n_params > 0:
                raise ValueError(
                    "Cannot mix Parameters and Modules in a sequence."
                )
        # Handle singular parameters and modules
        if isinstance(value, Parameter):
            self._parameters[key] = value
        elif isinstance(value, Module):
            self._modules[key] = value
        object.__setattr__(self, key, value)
```


Finally, we can implement a method that recursively visits all of the submodules and collects the registered parameters.

```python 
# firefly/nn/module.py

class Tensor: 
    
    ...

    def get_parameters(self, discard_non_trainable: bool = True) -> list[Parameter]:
        self._setup()
        parameters = []
        for parameter in self._parameters.values():
            if not isinstance(parameter, Parameter):
                for p in parameter:
                    if not discard_non_trainable or p.trainable:
                        parameters.append(p)
            elif not discard_non_trainable or parameter.trainable:
                parameters.append(parameter)
        for module in self._modules.values():
            if not isinstance(module, Module):
                for m in module:
                    parameters.extend(m.get_parameters(discard_non_trainable))
            else:
                parameters.extend(module.get_parameters(discard_non_trainable))
        return list(set((parameters)))
```

### Optional

At this point our `Module` is fully prepared for the training phase, but there are still a couple of features we can implement just so it is a bit more complete.

First, let's add the possibility of storing and loading the state dictionary of the module. The structure of the state dictionary is going to be the same as the one used in PyTorch: a key is a string that has the same name as the attribute of the module, if the parameter belongs to a submodule, the key is going to be a dot-separated string of the names of the submodules and the parameter itself (forming a path to the parameter). If any of the elements in the path are sequences, the position in the sequence is added to the path as well.

```python
# firefly/nn/module.py

class Tensor:
    ...

    def _gather_state_dict(self, prefix: str = "") -> dict:
        self._setup()
        state_dict = {}
        for key, parameter in self._parameters.items():
            full_key = f"{prefix}{key}"
            if isinstance(parameter, Parameter):
                state_dict[full_key] = parameter.get_state()
            else:
                for i, p in enumerate(parameter):
                    state_dict[f"{full_key}.{i}"] = p.get_state()
        for key, module in self._modules.items():
            full_key = f"{prefix}{key}."
            if isinstance(module, Module):
                state_dict.update(module._gather_state_dict(full_key))
            else:
                for i, m in enumerate(module):
                    state_dict.update(m._gather_state_dict(f"{full_key}.{i}."))
        return state_dict

    def load_state_dict(self, state_dict: dict, prefix: str = ""):
        self._setup()
        for key, value in state_dict.items():
            if not key.startswith(prefix):
                continue
            subkey = key[len(prefix) :]
            if "." in subkey:
                main_key, sub_subkey = subkey.split(".", 1)
                if main_key in self._modules:
                    module = self._modules[main_key]
                    if isinstance(module, Module):
                        module.load_state_dict(state_dict, f"{prefix}{main_key}.")
                    else:
                        index, sub_subkey = sub_subkey.split(".", 1)
                        module[int(index)].load_state_dict(
                            state_dict, f"{prefix}{main_key}.{index}."
                        )
                elif main_key in self._parameters:
                    param = self._parameters[main_key]
                    if isinstance(param, Parameter):
                        param.set_state(value)
                    else:
                        index = int(sub_subkey)
                        param[index].set_state(value)
            else:
                if subkey in self._parameters:
                    param = self._parameters[subkey]
                    if isinstance(param, Parameter):
                        param.set_state(value)
                    else:
                        for p in param:
                            p.set_state(value)

    def state_dict(self) -> dict:
        return self._gather_state_dict()
```

The second optional feature is the ability to enable/disable the training mode of the module. This is useful when we want to use the module for the inference only, as it allows to skip the persistence of the intermediate values required for the backward pass.

In order to switch the module to the training mode, we need to set the `requires_grad` attribute of all the parameters to `True`. Similarly, to switch the module to the evaluation mode, we need to set the `requires_grad` attribute of all the parameters to `False`.

```python

# firefly/nn/module.py

class Tensor:
    ...

    def train_mode(self):
        for param in self.get_parameters(True):
            param.tensor.requires_grad = True
    
    def eval_mode(self):
        for param in self.get_parameters(True):
            param.tensor.requires_grad = False
```

