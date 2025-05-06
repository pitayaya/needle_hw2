from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
from ..init.init_basic import ones_like, zeros_like
from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, axis=1, keepdims=True)  # 沿axis=1取最大值
        shifted_Z = Z - max_Z                             # 数值稳定偏移
        exp_Z = array_api.exp(shifted_Z)                  # 安全计算exp
        sum_exp_Z = array_api.sum(exp_Z, axis=1, keepdims=True)
        log_softmax_Z = shifted_Z - array_api.log(sum_exp_Z)  # 直接计算log(sum_exp)
        return log_softmax_Z
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]  # 获取前向传播的输入
        
        # 数值稳定的softmax计算
        max_Z = array_api.max(Z.numpy(), axis=-1, keepdims=True)  # 沿最后一个维度取最大值
        shifted_Z = Z.numpy() - max_Z
        exp_Z = array_api.exp(shifted_Z)
        sum_exp_Z = array_api.sum(exp_Z, axis=-1, keepdims=True)
        softmax_Z = exp_Z / sum_exp_Z
        
        # 正确的LogSoftmax梯度公式
        grad = out_grad.numpy() - (array_api.sum(out_grad.numpy(), axis=-1, keepdims=True) * softmax_Z)
        
        return (Tensor(grad),)
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        shifted_Z = Z - max_Z
        exp_Z = array_api.exp(shifted_Z) 
        sum_exp_Z = array_api.sum(exp_Z, axis=self.axes)
        log_sum_exp_Z = array_api.log(sum_exp_Z)
        max_Z = array_api.squeeze(max_Z, axis=self.axes)
        return  log_sum_exp_Z + max_Z
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            self.axes = tuple(range(len(node.inputs[0].shape)))
        z = node.inputs[0]
        shape = [1 if i in self.axes else z.shape[i] for i in range(len(z.shape))]
        gradient = exp(z - node.reshape(shape).broadcast_to(z.shape))
        return (out_grad.reshape(shape).broadcast_to(z.shape)*gradient,)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

