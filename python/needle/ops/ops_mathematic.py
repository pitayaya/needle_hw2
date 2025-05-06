"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
from ..init.init_basic import ones_like, zeros_like

import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad,)


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad_a = b * array_api.power(a, b - 1)
        grad_b = array_api.power(a, b) * log(a)
        return out_grad * grad_a, out_grad * grad_b
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        grad = self.scalar * array_api.power(a, self.scalar - 1)
        return (out_grad * grad,)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad_a = array_api.divide(ones_like(b), b)
        grad_b = -array_api.divide(a, array_api.power(b, 2))
        return out_grad * grad_a, out_grad * grad_b
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        grad = array_api.divide(1, self.scalar)
        return (out_grad * grad,)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            # 默认交换最后两个维度
            axes = list(range(a.ndim))
            axes[-1], axes[-2] = axes[-2], axes[-1]
            return array_api.transpose(a, axes=axes)
        
        # 处理部分axes指定的情况
        full_axes = list(range(a.ndim))
        if len(self.axes) == 2:
            # 只交换指定的两个维度
            i, j = self.axes
            full_axes[i], full_axes[j] = full_axes[j], full_axes[i]
        else:
            # 完全指定的axes
            assert len(self.axes) == a.ndim, f"Axes length {len(self.axes)} does not match input shape {a.shape}."
            full_axes = self.axes
            
        return array_api.transpose(a, axes=full_axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            # 默认交换最后两个维度
            return (transpose(out_grad),)
        
        # 构建逆映射：将转置后的轴顺序恢复为原始顺序
        input_shape = node.inputs[0].shape
        reverse_axes = list(range(len(input_shape)))
        
        # 处理部分轴交换的情况（如 axes=(1,2)）
        if len(self.axes) == 2:
            i, j = self.axes
            reverse_axes[i], reverse_axes[j] = reverse_axes[j], reverse_axes[i]
        else:
            # 处理完全指定的轴顺序
            reverse_axes = [0] * len(self.axes)
            for new_pos, old_pos in enumerate(self.axes):
                reverse_axes[old_pos] = new_pos
        
        # 再次转置以恢复原始形状
        return (transpose(out_grad, axes=tuple(reverse_axes)),)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return (reshape(out_grad, a.shape),)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        a_shape_aligned = [1] * (len(self.shape) - len(a.shape)) + list(a.shape)
        sum_axes = []
        for i, (dim_a, dim_out) in enumerate(zip(a_shape_aligned, self.shape)):
            if dim_a == 1 and dim_out > 1:
                sum_axes.append(i)
        if sum_axes:
            grad = summation(out_grad, tuple(sum_axes))
        else:
            grad = out_grad
        return (reshape(grad, a.shape),)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        out_shape = list(a.shape)
        if self.axes is None:
            out_shape = [1] * len(out_shape)
        else:
            for ax in self.axes:
                out_shape[ax] = 1
        return (broadcast_to(reshape(out_grad, out_shape), a.shape),)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad_a = matmul(out_grad, array_api.transpose(b))
        grad_b = matmul(array_api.transpose(a), out_grad)
        extra_dims_a = len(grad_a.shape) - len(a.shape)
        if extra_dims_a > 0:
            grad_a = summation(grad_a, tuple(range(extra_dims_a)))
        extra_dims_b = len(grad_b.shape) - len(b.shape)
        if extra_dims_b > 0:
            grad_b = summation(grad_b, tuple(range(extra_dims_b)))
        return grad_a, grad_b
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (-out_grad,)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return (out_grad / a,)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return (out_grad * exp(a),)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        mask = (a.realize_cached_data() > 0).astype(a.dtype)
        return (out_grad * mask,)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)




