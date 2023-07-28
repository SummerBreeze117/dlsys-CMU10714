"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


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
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


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


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return a ** (self.scalar - 1) * self.scalar * out_grad
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return (out_grad / rhs, - lhs * out_grad / rhs ** 2)  
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes:
            ax0, ax1 =  self.axes[0], self.axes[1]
        else:
            ax0, ax1 = a.ndim - 2, a.ndim - 1
        new_axes = list(range(a.ndim))
        new_axes[ax0], new_axes[ax1] = new_axes[ax1], new_axes[ax0]
        return a.permute(new_axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ori_shape = node.inputs[0].shape
        return out_grad.reshape(ori_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ori_shape = node.inputs[0].shape
        if ori_shape == self.shape:
            return out_grad

        # shrink_dims = list(range(len(self.shape)))
        # iterate from the back because it could be len(ori_shape) < len(self.shape)
        # for i, (ori, cur) in enumerate(zip(reversed(ori_shape), reversed(self.shape))):
        #     if ori == cur:
        #         shrink_dims[len(self.shape) - i - 1] = -1
        # shrink_dims = tuple(filter(lambda x: x >= 0, shrink_dims))
        # return out_grad.sum(shrink_dims).reshape(ori_shape)
        shrink_dims = []
        for i, (ori, cur) in enumerate(zip(ori_shape, self.shape)):
            if ori != cur:
                shrink_dims.append(i)
        return out_grad.sum(shrink_dims).reshape(ori_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if isinstance(self.axes, (list, tuple)) and len(self.axes) > 1:
            # multiple axes case
            for axis in reversed(sorted(self.axes)):
                a = a.sum(axis = axis)
            return a
        return array_api.summation(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        new_shape = list(node.inputs[0].shape)
        if self.axes is None:
            axes = range(len(new_shape))
        elif isinstance(self.axes, tuple):
            axes = self.axes
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            raise ValueError("Unsupported axes type, must be int, tuple or None!")
        
        for axis in axes:
            new_shape[axis] = 1
        return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lgrad, rgrad = out_grad @ rhs.transpose(), lhs.transpose() @ out_grad

        # X(B, M, K) matmul W(K, N)
        # out(B, M, N)
        # dX = out matmul W_transpose
        # dW = X_transpose matmul out -> B, K, N，然后在B维度上reduce-> K, N

        if len(lhs.shape) < len(lgrad.shape):
            lgrad = lgrad.sum(tuple(range(len(lgrad.shape) - len(lhs.shape))))
        if len(rhs.shape) < len(rgrad.shape):
            rgrad = rgrad.sum(tuple(range(len(rgrad.shape) - len(rhs.shape))))
        return (lgrad, rgrad)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return - a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return - out_grad
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
        return out_grad / a
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
        return out_grad * exp(a)
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
        a = node.inputs[0].realize_cached_data()
        return out_grad * Tensor(a > 0, device=out_grad.device, dtype=out_grad.dtype, requires_grad=False)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        z_max_origindim = Z.max(self.axes, keepdims=True)
        z_max_reducedim = Z.max(self.axes)
        return array_api.log(array_api.exp(Z - z_max_origindim.broadcast_to(Z.shape)).sum(self.axes)) + z_max_reducedim
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        max_z = z.realize_cached_data().max(self.axes, keepdims=True)
        exp_z = array_api.exp(z.realize_cached_data() - max_z.broadcast_to(z.shape))
        sum_exp_z = exp_z.sum(self.axes)
        grad_sum_exp_z = out_grad / sum_exp_z

        #   [2.132311 4.359107 2.888913] ->
        #  [[[2.132311 4.359107 2.888913]
        #   [2.132311 4.359107 2.888913]
        #   [2.132311 4.359107 2.888913]]
        #
        #  [[2.132311 4.359107 2.888913]
        #   [2.132311 4.359107 2.888913]
        #   [2.132311 4.359107 2.888913]]
        #
        #  [[2.132311 4.359107 2.888913]
        #   [2.132311 4.359107 2.888913]
        #   [2.132311 4.359107 2.888913]]]
        if self.axes is None:
            return grad_sum_exp_z.broadcast_to(z.shape) * exp_z

        #  self.axes = (1, 2)
        # [2.132311 4.359107 2.888913] ->
        # [[[2.132311 2.132311 2.132311]
        #   [2.132311 2.132311 2.132311]
        #  [2.132311 2.132311 2.132311]]
        #
        # [[4.359107 4.359107 4.359107]
        #  [4.359107 4.359107 4.359107]
        #  [4.359107 4.359107 4.359107]]
        #
        # [[2.888913 2.888913 2.888913]
        #  [2.888913 2.888913 2.888913]
        #  [2.888913 2.888913 2.888913]]]
        expand_shape = list(z.shape)
        for axis in self.axes:
            expand_shape[axis] = 1
        grad_exp_z = grad_sum_exp_z.reshape(expand_shape).broadcast_to(z.shape)
        return grad_exp_z * exp_z
        ## END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * (1 - tanh(a) ** 2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        assert len(args) > 0, "Stack needs at least one array!"
        shape = args[0].shape
        for a in args:
            assert shape == a.shape, "All arrays need to be of the same size!"
        new_shape = list(shape)
        new_shape.insert(self.axis, len(args))
        out = args[0].device.empty(new_shape)

        slices = [slice(0, s) for s in new_shape]
        for i, arg in enumerate(args):
            slices[self.axis] = slice(i, i + 1)
            out[tuple(slices)] = arg
        return out
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        
        out = []
        slices = [slice(0, s) for s in A.shape]
        for i in range(A.shape[self.axis]):
            slices[self.axis] = slice(i, i + 1)
            out.append(A[tuple(slices)].compact().reshape(tuple(new_shape)))
        return tuple(out)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        for axis in self.axes:
            new_shape[axis] *= self.dilation + 1
        new_shape = tuple(new_shape)
        out = array_api.full(new_shape, 0, device=a.device)
        slices = [slice(0, s) for s in new_shape]
        for axis in self.axes:
            slices[axis] = slice(0, new_shape[axis], self.dilation + 1)
        out[tuple(slices)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        slices = [slice(0, s) for s in a.shape]
        for axis in self.axes:
            slices[axis] = slice(0, a.shape[axis], self.dilation + 1)
        out = a[tuple(slices)]
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        N, H, W, C_in = A.shape
        K, K_, C_in_, C_out = B.shape
        assert K == K_
        Ns, Hs, Ws, Cs = A.strides

        H_out, W_out = (H - K + 1) // self.stride, (W - K + 1) // self.stride
        img2col = A.as_strided(shape=(N, H_out, W_out, K, K, C_in),
                         strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs))\
                         .compact().reshape((N * H_out * W_out, K * K * C_in))

        out = img2col @ B.compact().reshape((K * K_ * C_in_, C_out))
        return out.reshape((N, H_out, W_out, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs
        K, _, _, _ = W.shape

        # out_grad: # N * ((H-K+1+2P)/self.stride) * ((W-K+1+2P)/self.stride) * C_out

        out_grad = dilate(out_grad, (1, 2), self.stride - 1) # N * (H-K+1+2P) * (W-K+1+2P) * C_out
        W_T = transpose(flip(W, (0, 1)), (2, 3))  # K * K * C_out * C_in
        X_grad = conv(out_grad, W_T, padding= K - 1 - self.padding)
        
        # The gradients of W must be accumulated over the batches.
        # Consider turning batches into channels to make the conv operator itself do this accumulation.
        X_permute = transpose(X, (0, 3))  # C_in * H * W * N
        grad_permute = transpose(transpose(out_grad, (0, 1)), (1, 2))  # (H-K+1+2P) * (W-K+1+2P) * N * C_out
        W_grad = conv(X_permute, grad_permute, padding=self.padding)  # C_in * H * W * C_out
        W_grad = transpose(transpose(W_grad, (0, 1)), (1, 2))  # K * K * C_in * C_out

        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



