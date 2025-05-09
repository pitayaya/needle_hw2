"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).reshape((1, out_features)))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output = ops.matmul(X, self.weight)

        if self.bias is not None:
            bias_expanded = self.bias.broadcast_to(output.shape)
            output += bias_expanded

        return output
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        total_x = 1
        for i in range(1, len(X.shape)):
            total_x *= X.shape[i]
        return ops.reshape(X, (X.shape[0], total_x))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module.forward(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        log_sum_exp = ops.logsumexp(logits, axes=(1,))
        one_hot_labels = init.one_hot(logits.shape[1], y)
        true_one_hot_labels = ops.summation(logits * one_hot_labels, axes=(1,))
        loss = log_sum_exp - true_one_hot_labels
        avg_loss = ops.summation(loss, axes=(0,)) / logits.shape[0]
        return avg_loss
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype), device=device, dtype=dtype)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        weight_broadcast = ops.broadcast_to(self.weight, x.shape)
        bias_broadcast = ops.broadcast_to(self.bias, x.shape)
        if self.training:
            mean_x = ops.summation(x, axes=(0,)) / x.shape[0]
            # print("Training mode: mean_x =", mean_x)
            self.running_mean = ((1 - self.momentum) * self.running_mean + self.momentum * mean_x).detach()
            mean_x = ops.broadcast_to(ops.reshape(mean_x, (1, x.shape[1])), x.shape)
            var_x = ops.summation(ops.power_scalar(x - mean_x, 2), axes=(0,)) / x.shape[0]
            # print("Training mode: var_x =", var_x)
            self.running_var = ((1 - self.momentum) * self.running_var + self.momentum * var_x).detach()
            var_x = ops.broadcast_to(ops.reshape(var_x, (1, x.shape[1])), x.shape)
            normalized = (x - mean_x) / ops.power_scalar(var_x + self.eps, 0.5)
            # print("Normalized (train mode):", normalized)
            return weight_broadcast * normalized + bias_broadcast
        else:
            normalized = (x - self.running_mean.broadcast_to(x.shape)) / ops.power_scalar(self.running_var.broadcast_to(x.shape) + self.eps, 0.5)
            # print("Inference mode: normalized =", normalized)
            return weight_broadcast * normalized + bias_broadcast

        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean_x = ops.summation(x, axes=(1,)) / x.shape[1]
        mean_x = ops.broadcast_to(ops.reshape(mean_x, (x.shape[0], 1)), x.shape)
        var_x = ops.summation(ops.power_scalar(x - mean_x, 2), axes=(1,)) / x.shape[1]
        var_x = ops.broadcast_to(ops.reshape(var_x, (x.shape[0], 1)), x.shape)
        weight_broadcast = ops.broadcast_to(self.weight, x.shape)
        bias_broadcast = ops.broadcast_to(self.bias, x.shape)
        normalized = (x - mean_x) / ops.power_scalar(var_x + self.eps, 0.5)
        return weight_broadcast * normalized + bias_broadcast
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            drop_matrix = init.randb(*x.shape, p=(1-self.p))
            return x * drop_matrix / (1-self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
