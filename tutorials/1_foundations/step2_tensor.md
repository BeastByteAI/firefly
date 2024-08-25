# Tensor


In the previous chapter we implemented several differentiable operators that work over numpy arrays. Now we need to create a `Tensor` class that would be responsible for applying and keeping track of the operators.


```python 
# firefly/tensor.py
from __future__ import annotations
import numpy as np
from firefly.ops.base import BaseOp
from firefly.ops import Mul, MatMul, Add, Sub, Div, ReLU, ReduceSum
from typing import Type

class Tensor:

    def __init__(
        self,
        data: np.ndarray,
        requires_grad=False,
        store_grad_non_leaf=False,
    ):
        if isinstance(data, float):
            data = np.array([data], dtype=np.float32)
        elif isinstance(data, int):
            data = np.array([data], dtype=np.int64)
        self.buffer = np.asarray(data)
        self.requires_grad = requires_grad
        self.store_grad_non_leaf = store_grad_non_leaf
        self.grad = None
        self.creator: BaseOp | None = None
        self.parents: tuple[Tensor] | None = None 

    def __repr__(self) -> str:
        return f"Tensor({self.buffer}, shape = {self.buffer.shape}, requires_grad = {self.requires_grad})"
```

Our tensor will be storing the following information: 
- `buffer`: the underlying (device-specific) data buffer. For now, we will just use a np.array as a buffer;
- `requires_grad`: a boolean flag indicating whether the tensor requires the gradient during the backward pass;
- `store_grad_non_leaf`: indicates whether the gradient must be stored for non-leaf tensors (in most of the cases we would not need that);
- `grad`: the gradient value to be set during the backward pass;
- `creator`: the operator that "created" the tensor;
- `parents`: the tensors that were inputs of the creator operator.


Before proceeding with the `Tensor` class, we also need to define a small utility function that applies an operator and wraps the result into a new tensor instance. 

```python
# firefly/tensor.py

def apply_op(op_class: Type[BaseOp], *args: Tensor, **kwargs):
    op = op_class()
    result_buffer = op.forward(*[arg.buffer for arg in args], **kwargs)
    result = Tensor(
        result_buffer,
        requires_grad=(any((arg.requires_grad for arg in args)) and op.differentiable),
    )
    if result.requires_grad:
        result.creator = op
        result.parents = args
    return result
```



Now we can define dunder methods that apply the corresponding operators. 

```python
class Tensor:
    ...

    def __matmul__(self, other: Tensor) -> Tensor:
        return apply_op(MatMul, self, other)

    def __mul__(self, other: Tensor) -> Tensor:
        return apply_op(Mul, self, other)

    def __add__(self, other: Tensor) -> Tensor:
        return apply_op(Add, self, other)

    def __sub__(self, other: Tensor) -> Tensor:
        return apply_op(Sub, self, other)

    def __truediv__(self, other: Tensor) -> Tensor:
        return apply_op(Div, self, other)

    def relu(self) -> Tensor:
        return apply_op(ReLU, self)

    def reduce_sum(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        return apply_op(ReduceSum, self, axis=axis, keepdims=keepdims)

    def to_numpy(self) -> np.ndarray:
        return self.buffer.copy()
```

We can check that our tensor works by performing a simple scalar computation. We will use multiplications only, but you can test it with other ops as well. 

```python 
# let a = 1, b = 2, w = 3
# c = b * w = 2 * 3 = 6
# d = a * w = 1 * 3 = 3
# e = c * d = 6 * 3 = 18

from firefly.tensor import Tensor

a = Tensor(1.)
b = Tensor(2.)
w = Tensor(3.)

c = b * w
d = a * w
e = c * d
print(e)
# Output: Tensor([18.], shape = (1,), requires_grad = False)
```

Right now our tensor is not yet suitable for the NN training, as it does not have an implementation of the backward pass. 


To fix that, we need to implement `backward()` method that
 - computes the gradients of itself wrt own inputs (using the backward method of the creator operator);
 - multiplies the computed gradients by the incoming gradient;
 - calls the `backward()` method of the parents and pass the respective gradients.

``` python 
class Tensor:
    ...

    def backward(self, grad_output=None):
        if not self.requires_grad:
            return

        if grad_output is None:
            if self.buffer.size == 1:
                grad_output = Tensor(1.0)
            else:
                raise RuntimeError("Grad must be specified for non-scalar tensors.")

        # usually we only need the grads for the leaf tensors
        if not self.parents or self.store_grad_non_leaf:
            if self.grad is None:
                self.grad = grad_output
            else:
                self.grad += grad_output

        if self.creator and self.parents:
            grad_inputs = self.creator.backward(
                grad_output.buffer
            )  # only propagate new grad
            for inp, grad in zip(self.parents, grad_inputs):
                inp.backward(Tensor(grad))
    
    def zero_grad(self):
        self.grad = None
```

We can verify that this implementation works by taking the earlier example and calculating the gradient of `e` wrt `w`:

```python
# let a = 1, b = 2, w = 3
# c = b * w = 2 * 3 = 6
# d = a * w = 1 * 3 = 3
# e = c * d = 6 * 3 = 18


from firefly.tensor import Tensor

a = Tensor(1.)
b = Tensor(2.)
w = Tensor(3.)
w.requires_grad = True

c = b * w
d = a * w
e = c * d
print(e)
# Output: Tensor([18.], shape = (1,), requires_grad = False)


e.backward()

# de/dd = c = 6
# de/dc = d = 3
# de/dw = de/dc * dc/dw + de/dd * dd/dw = 3 * b + 6 * a = 3 * 2 + 6 * 1 = 12

print(w.grad)
# Output: Tensor([12.], shape = (1,), requires_grad = False)
```

We can also check that the outputs would be the same if we did the same computations with different popular frameworks:

```python
# PyTorch

import torch

a = torch.tensor(1., requires_grad=False)
b = torch.tensor(2., requires_grad=False)
w = torch.tensor(3., requires_grad=True)

c = b * w
d = a * w
e = c * d
print(e)

e.backward()
grad = w.grad
print(grad)
```


```python 
# TensorFlow

import tensorflow as tf

a = tf.constant(1., dtype=tf.float32)
b = tf.constant(2., dtype=tf.float32)
w = tf.Variable(3., dtype=tf.float32)


with tf.GradientTape() as tape:
    c = b * w
    d = a * w
    e = c * d
    print(e)

grad = tape.gradient(e, w)
print(grad)
```


```python
# JAX

import jax
import jax.numpy as jnp

a = jnp.array(1.)
b = jnp.array(2.)
w = jnp.array(3.)

def f(a, b, w):
    c = b * w
    d = a * w
    e = c * d
    return e

e = f(a, b, w)
print(e)

grad = jax.grad(f, argnums=2)(a, b, w)

print(grad)
```

