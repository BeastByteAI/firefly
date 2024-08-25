# Preface

The goal of this tutorial is to build an initial version of a learning framework from scratch without using any external libraries except numpy. The tutorial is divided into 4 steps and ends with an example of training a simple neural network. Each of the step may take between 30 to 120 minutes to complete, depending on your familiarity with the concepts and how deeply you want to understand the implementation details. However, it is recommended not to dive too deep into the details on the first pass, and instead focus on understanding the overall structure and how the components interact with each other. Once you have a good grasp of the overall structure, you can revisit the tutorial to understand the details.

The tutorial is designed to be beginner-friendly, but does not cover too much theory. If you are unfamiliar with the basics of deep learning, it is recommended to check out some introductory materials first. For example, [the first lecture of MIT 6.S191](https://www.youtube.com/watch?v=JN6H4rQvwgY).

If you have any questions or suggestions, feel free to contact the authors via [Discord](https://discord.com/invite/YDAbwuWK7V) or GitHub.

# Introduction

Deep learning frameworks are powerful tools that enable researchers and developers to design, train, and deploy neural networks with relative ease. These frameworks handle a variety of complex tasks behind the scenes, from automatic differentiation to GPU acceleration, allowing users to focus more on modeling and less on algorithmic details. However, the inner workings of these frameworks are often overlooked, as users tend to take their functionality for granted without considering how they are implemented. In this series of tutorials, we aim to shed light on the core components of deep learning frameworks by building (an oversimplified version of) one from scratch.

Before starting the implementation, let's try to answer the following question: "What is the main feature of a deep learning framework?"

There can be different answers to this question, but it is safe to say that at least one of the main features would be the ability to define an arbitrary neural net architecture. In order to do so, every neural net should be decomposable into a set of reusable building blocks.

For example, almost every DL course starts with teaching that neural nets consist of layers (where layers can optionally be groupped into blocks). While correct, this level of abstraction is a bit too high for our purposes. After all, we should be able to use the framework in order to easily define new layers as well, while receiving the automatic differentiation out of the box. 

Therefore, it is necessary to introduce two additional (lower) levels of abstractions:
- Functions / Operators: these are the computational blocks for which both forward and (when applicable) backward passes are defined. Every higher level component is a composition of functions and hence the gradients can be computed using the chain rule.
- Primitives: these are the lowest level computational blocks which have backend-specific implementations. Both forward and backward passes of the functions consist of one or several primitives. By having the dedicated primitives we can easily introduce new backends while retaining all of the higher level components. For example, imagine that we have a CPU only implementation of the framework. In this case, the only thing we have to do in order to add the CUDA support is to implement the primitives, while the functions and layers can be used as-is due to them being backend-agnostic.

<picture>
  <img alt="Levels" src="https://gist.githubusercontent.com/OKUA1/913a22e9ad668b46698303bcc4495ceb/raw/fc67a868357ae8db67b826ac0196c063974734ed/levels.svg" height = "550">
</picture>

# Operators

For now, for the sake of simplicity, we can start with a single-backend implementation and hence omit the primitives. 

We begin the implementation with the base operator/function and two subclasses for differentiable and non-differentiable versions respectively. 

```python
# firefly/ops/base.py

import numpy as np
from abc import ABC, abstractmethod


class BaseOp(ABC):

    differentiable: bool

    @abstractmethod
    def forward(self, *buffers: np.ndarray, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        # the backward returns a tuple of grads corresponding to each input
        pass


class DiffOp(BaseOp):

    differentiable = True

    def maybe_debroadcast_grad(self, grad: np.ndarray, input: np.ndarray) -> np.ndarray:
        input_shape = input.shape
        grad_shape = grad.shape

        if input_shape == grad_shape:
            return grad

        ndims_grad = len(grad_shape)
        ndims_input = len(input_shape)
        # remove the prepended dims
        if ndims_grad > ndims_input:
            grad = np.sum(
                grad, axis=tuple(range(ndims_grad - ndims_input)), keepdims=False
            )

        grad_shape = grad.shape
        reduction_axes = []
        # reduce the repeated dims
        for i in range(len(grad_shape)):
            if grad_shape[i] > input_shape[i]:
                reduction_axes.append(i)

        if reduction_axes:
            grad = np.sum(grad, axis=tuple(reduction_axes), keepdims=True)

        return grad


class NonDiffOp(BaseOp):

    differentiable = False

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        raise RuntimeError("Called `backward()` of non-differentiable operation")


```

For the differentiable operator we add a utlility method `maybe_debroadcast_grad`. We are going to build the framework based on the assumption that the computation backend can automatically perform the [numpy-style input broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html). Therefore, during the backward pass, we might need to debroadcast the gradients to the original input shapes. 


Next, we can start implementing some simple operators like `Mul`, `Add`, `Sub`, `Div`, `MatMul`, `ReLU`, `ReduceMean`. These should be sufficient to build a basic multilayer network later.

```python
# firefly/ops/add.py

from firefly.ops.base import DiffOp
import numpy as np


class Add(DiffOp):
    def forward(self, *buffers: np.ndarray, **kwargs) -> np.ndarray:
        assert len(buffers) == 2, "Add requires exactly 2 inputs"
        self.buffers = buffers
        return buffers[0] + buffers[1]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad_a = self.maybe_debroadcast_grad(grad, self.buffers[0])
        grad_b = self.maybe_debroadcast_grad(grad, self.buffers[1])
        return (grad_a, grad_b)
```

```python
# firefly/ops/mul.py

from firefly.ops.base import DiffOp
import numpy as np


class Mul(DiffOp):
    def forward(self, *buffers: np.ndarray, **kwargs) -> np.ndarray:
        assert len(buffers) == 2, "Mul requires exactly 2 inputs"
        self.inputs = buffers
        return buffers[0] * buffers[1]

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.inputs
        grad_a = self.maybe_debroadcast_grad(grad * b, a)
        grad_b = self.maybe_debroadcast_grad(grad * a, b)
        return (grad_a, grad_b)
```

```python
# firefly/ops/div.py 

from firefly.ops.base import DiffOp
import numpy as np


class Div(DiffOp):
    def forward(self, *buffers: np.ndarray, **kwargs) -> np.ndarray:
        assert len(buffers) == 2, "Div requires exactly 2 inputs"
        self.buffers = buffers
        return buffers[0] / buffers[1]

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.buffers
        grad_a = grad / b  # grad * (1/x2)
        grad_b = grad * (-a / (b**2))  # grad * (-x1/x2^2)
        grad_a = self.maybe_debroadcast_grad(grad_a, a)
        grad_b = self.maybe_debroadcast_grad(grad_b, b)
        return (grad_a, grad_b)
```


```python 
# firefly/ops/sub.py

from firefly.ops.base import DiffOp
import numpy as np


class Sub(DiffOp):
    def forward(self, *buffers: np.ndarray, **kwargs) -> np.ndarray:
        assert len(buffers) == 2, "Sub requires exactly 2 inputs"
        self.buffers = buffers
        return buffers[0] - buffers[1]

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.buffers
        return (
            self.maybe_debroadcast_grad(grad, a),
            self.maybe_debroadcast_grad(-grad, b),
        )
```

```python 
# firefly/ops/matmul.py

from firefly.ops.base import DiffOp
import numpy as np

class MatMul(DiffOp):
    def forward(self, *buffers: np.ndarray, **kwargs) -> np.ndarray:
        assert len(buffers) == 2, "MatMul requires exactly 2 inputs"
        self.inputs = buffers
        result = self.inputs[0] @ self.inputs[1]
        return result

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.inputs

        axes_a = list(range(a.ndim))
        axes_a[-2], axes_a[-1] = axes_a[-1], axes_a[-2]
        a_transposed = np.transpose(a, axes=axes_a)

        axes_b = list(range(b.ndim))
        axes_b[-2], axes_b[-1] = axes_b[-1], axes_b[-2]
        b_transposed = np.transpose(b, axes=axes_b)

        grad_a = np.matmul(grad, b_transposed)
        grad_b = np.matmul(a_transposed, grad)

        return (
            self.maybe_debroadcast_grad(grad_a, a),
            self.maybe_debroadcast_grad(grad_b, b),
        )
```

```python
# firefly/ops/reduce_sum.py

import numpy as np
from firefly.ops.base import DiffOp

class ReduceSum(DiffOp):
    def forward(
        self, *buffers: np.ndarray, axis=None, keepdims=False, **kwargs
    ) -> np.ndarray:
        assert len(buffers) == 1, "ReduceSum requires exactly 1 input"
        self.buffer = buffers[0]
        self.axis = axis
        self.keepdims = keepdims
        return np.sum(buffers[0], axis=axis, keepdims=keepdims)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        input_shape = self.buffer.shape
        grad_shape = list(input_shape)

        if self.axis is None:
            grad_shape = [1] * len(input_shape)
        else:
            axes = [self.axis] if isinstance(self.axis, int) else self.axis
            for ax in axes:
                grad_shape[ax] = 1

        grad = np.reshape(grad, grad_shape)
        return (np.broadcast_to(grad, input_shape),)
```


```python 
# firefly/ops/__init__.py

from firefly.ops.add import Add
from firefly.ops.mul import Mul
from firefly.ops.relu import ReLU
from firefly.ops.sub import Sub
from firefly.ops.div import Div
from firefly.ops.matmul import MatMul
from firefly.ops.reduce_sum import ReduceSum
```