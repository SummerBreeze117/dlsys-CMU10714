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
        self.weight_decay = weight_decay  # for L2

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            # note: L2 Regularization is belong to ∇
            grad = param.grad.data + self.weight_decay * param.data

            if param in self.u:
                self.u[param] = self.momentum * self.u[param] + (1 - self.momentum) * grad
            else:
                self.u[param] = (1 - self.momentum) * grad

            param.data -= self.lr * self.u[param]
        ### END YOUR SOLUTION


class Adam(Optimizer):
    """
    Implements Adam algorithm, proposed in https://arxiv.org/abs/1412.6980
    """
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
            # note: L2 Regularization is belong to ∇
            grad = param.grad.data + self.weight_decay * param.data

            if param in self.m:
                self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
            else:
                self.m[param] = (1 - self.beta1) * grad
            if param in self.v:
                self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * grad ** 2
            else:
                self.v[param] = (1 - self.beta2) * grad ** 2

            m_bar = self.m[param] / (1 - self.beta1 ** self.t)
            v_bar = self.v[param] / (1 - self.beta2 ** self.t)

            param.data -= self.lr * m_bar / (v_bar ** 0.5 + self.eps)
        ### END YOUR SOLUTION
