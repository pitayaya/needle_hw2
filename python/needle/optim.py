"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if param.grad is None:
                continue

            grad = (param.grad.data + self.weight_decay * param.data).detach()

            if self.momentum > 0:
                if param not in self.u:
                    self.u[param] = ndl.zeros_like(param.data)
                self.u[param] = self.momentum * self.u[param] + (1 - self.momentum) * grad
                update = self.u[param]
            else:
                update = grad

            param.data -= ndl.Tensor(self.lr * update, dtype=param.data.dtype)
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        total_norm = 0.0

        for param in self.params:
            if param.grad is not None:
                total_norm += (param.grad.data ** 2).sum()
        
        total_norm = total_norm ** 0.5

        if total_norm > max_norm:
            for param in self.params:
                if param.grad is not None:
                    param.grad.data *= max_norm / total_norm
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1

        for param in self.params:
            if param.grad is None:
                continue

            grad = param.grad.data + self.weight_decay * param.data

            if param not in self.m:
                self.m[param] = ndl.zeros_like(param.data)
                self.v[param] = ndl.zeros_like(param.data)
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[param] / (1 - (self.beta1 ** self.t))
            v_hat = self.v[param] / (1 - (self.beta2 ** self.t))
            param.data -= ndl.Tensor(self.lr * m_hat / (v_hat ** 0.5 + self.eps), dtype=param.data.dtype)
            

        ### END YOUR SOLUTION
