import unittest
import torch
import numpy as np
from firefly.tensor import Tensor

backend = "cpu"


class TestTensorOperations(unittest.TestCase):

    def assert_tensors_close(
        self, t1: Tensor, t2: torch.Tensor, atol=1e-5, op_name=None
    ):
        self.assertTrue(
            np.allclose(t1.to_numpy(), t2.detach().numpy(), atol=atol),
            f"{op_name} :: Tensors are not close: firefly:\n{t1.to_numpy()} vs torch: \n{t2.detach().numpy()}",
        )

    def assert_grads_close(self, t1: Tensor, t2: torch.Tensor, atol=1e-5, op_name=None):
        if t1.grad is None or t2.grad is None:
            return
        self.assertTrue(
            np.allclose(t1.grad.to_numpy(), t2.grad.detach().numpy(), atol=atol),
            f"{op_name} :: Gradients are not close: {t1.grad.to_numpy()} vs {t2.grad.detach().numpy()}",
        )

    def run_test(
        self, data1, data2, func_firefly, func_torch, backward=True, op_name=None
    ):
        torch_dtype1 = torch.float32 if data1.dtype == np.float32 else torch.int64
        torch_dtype2 = torch.float32 if data2.dtype == np.float32 else torch.int64
        requires_grad2 = torch.dtype == torch.float32
        t1 = Tensor(data1, requires_grad=True).to(backend)
        t2 = torch.tensor(data1, dtype=torch_dtype1, requires_grad=True)
        result1: Tensor = func_firefly(
            t1, Tensor(data2, requires_grad=requires_grad2).to(backend)
        )
        result2 = func_torch(
            t2, torch.tensor(data2, dtype=torch_dtype2, requires_grad=requires_grad2)
        )
        self.assert_tensors_close(result1, result2, 1e-5, op_name)
        if backward and t1.requires_grad:
            grad_output = torch.ones_like(result2)
            result2.backward(grad_output)
            result1.backward(Tensor(grad_output.numpy()).to(backend))
            self.assert_grads_close(result1, result2, op_name)

    def test_add(self):
        data1 = np.random.randn(2, 3).astype(np.float32)
        data2 = np.random.randn(2, 3).astype(np.float32)
        self.run_test(
            data1,
            data2,
            lambda x, y: x + y,
            lambda x, y: x + y,
            op_name="ADD",
            backward=True,
        )

    def test_sub(self):
        data1 = np.random.randn(2, 3).astype(np.float32)
        data2 = np.random.randn(2, 3).astype(np.float32)
        self.run_test(
            data1,
            data2,
            lambda x, y: x - y,
            lambda x, y: x - y,
            op_name="SUB",
            backward=True,
        )

    def test_mul(self):
        data1 = np.random.randn(2, 3).astype(np.float32)
        data2 = np.random.randn(2, 3).astype(np.float32)
        self.run_test(
            data1,
            data2,
            lambda x, y: x * y,
            lambda x, y: x * y,
            op_name="MUL",
            backward=True,
        )

    def test_div(self):
        data1 = np.random.randn(2, 3).astype(np.float32)
        data2 = (
            np.random.randn(2, 3).astype(np.float32) + 1e-6
        )  # to avoid division by zero
        self.run_test(
            data1,
            data2,
            lambda x, y: x / y,
            lambda x, y: x / y,
            op_name="DIV",
            backward=True,
        )

    def test_matmul(self):
        data1 = np.random.randn(2, 3).astype(np.float32)
        data2 = np.random.randn(3, 4).astype(np.float32)
        self.run_test(
            data1,
            data2,
            lambda x, y: x @ y,
            lambda x, y: x @ y,
            op_name="MATMUL",
            backward=True,
        )

    def test_relu(self):
        data = np.random.randn(2, 3).astype(np.float32)
        self.run_test(
            data,
            data,
            lambda x, _: x.relu(),
            lambda x, _: x.relu(),
            backward=True,
            op_name="RELU",
        )

    def test_sigmoid(self):
        data = np.random.randn(2, 3).astype(np.float32)
        self.run_test(
            data,
            data,
            lambda x, _: x.sigmoid(),
            lambda x, _: torch.sigmoid(x),
            backward=True,
            op_name="SIGMOID",
        )

    def test_reduce_sum(self):
        data = np.random.randn(2, 3, 4).astype(np.float32)
        self.run_test(
            data,
            data,
            lambda x, _: x.reduce_sum(axis=1, keepdims=True),
            lambda x, _: torch.sum(x, dim=1, keepdim=True),
            backward=True,
            op_name="REDUCE_SUM",
        )

    def test_reduce_mean(self):
        data = np.random.randn(2, 3, 4).astype(np.float32)
        self.run_test(
            data,
            data,
            lambda x, _: x.reduce_mean(axis=1, keepdims=True),
            lambda x, _: torch.mean(x, dim=1, keepdim=True),
            backward=False,
            op_name="REDUCE_MEAN",
        )

    def test_reshape(self):
        data = np.random.randn(2, 3, 4).astype(np.float32)
        self.run_test(
            data,
            data,
            lambda x, _: x.reshape((3, 8)),
            lambda x, _: x.reshape((3, 8)),
            backward=True,
            op_name="RESHAPE",
        )

    def test_sqrt(self):
        data = np.abs(np.random.randn(2, 3)).astype(np.float32)
        self.run_test(
            data,
            data,
            lambda x, _: x.sqrt(),
            lambda x, _: torch.sqrt(x),
            backward=True,
            op_name="SQRT",
        )

    def test_transpose(self):
        data = np.random.randn(2, 3, 4).astype(np.float32)
        self.run_test(
            data,
            data,
            lambda x, _: x.transpose((1, 0, 2)),
            lambda x, _: x.permute((1, 0, 2)),
            backward=True,
            op_name="TRANSPOSE",
        )

    def test_softmax(self):
        data = np.random.randn(2, 3).astype(np.float32)
        self.run_test(
            data,
            data,
            lambda x, _: x.softmax(axis=1),
            lambda x, _: torch.softmax(x, dim=1),
            backward=True,
        )

    def test_exp(self):
        data = np.random.randn(2, 3).astype(np.float32)
        self.run_test(
            data, data, lambda x, _: x.exp(), lambda x, _: torch.exp(x), backward=True
        )

    def test_tanh(self):
        data = np.random.randn(2, 3).astype(np.float32)
        self.run_test(
            data,
            data,
            lambda x, _: x.tanh(),
            lambda x, _: torch.tanh(x),
            backward=True,
        )

    def test_slice(self):
        data = np.random.randn(4, 5).astype(np.float32)
        data1 = np.random.randn(4, 5, 5).astype(np.float32)
        self.run_test(
            data,
            data,
            lambda x, _: x[1:3, 2:4],
            lambda x, _: x[1:3, 2:4],
            backward=True,
            op_name="SLICE",
        )
        self.run_test(
            data1,
            data1,
            lambda x, _: x[1:3, 2:4],
            lambda x, _: x[1:3, 2:4],
            backward=True,
            op_name="SLICE",
        )

    def test_concat(self):
        data1 = np.random.randn(2, 3).astype(np.float32)
        data2 = np.random.randn(2, 3).astype(np.float32)
        self.run_test(
            data1,
            data2,
            lambda x, y: x.concat([y], axis=1),
            lambda x, y: torch.cat((x, y), dim=1),
            op_name="CONCAT",
            backward=True,
        )

    def test_gather_nd(self):
        data = np.random.randn(2, 3, 4).astype(np.float32)
        indices = np.array([[0, 0], [1, 2]])

        def gather_2d(x, indices):
            return x[indices[:, 0], indices[:, 1]]

        self.run_test(
            data,
            indices,
            lambda x, y: x.gather_nd(y),
            lambda x, y: gather_2d(x, y),
            backward=True,
        )


if __name__ == "__main__":
    unittest.main()
