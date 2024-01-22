# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Custom PyTorch ops for efficient bias and activation."""

import numpy as np
import torch
from typing import Any, List, Tuple, Union


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

# ----------------------------------------------------------------------------


activation_funcs = {
    'linear':   EasyDict(func=lambda x, **_:         x,                                          def_alpha=0,    def_gain=1,             cuda_idx=1, ref='',  has_2nd_grad=False),
    'relu':     EasyDict(func=lambda x, **_:         torch.nn.functional.relu(x),                def_alpha=0,    def_gain=np.sqrt(2),    cuda_idx=2, ref='y', has_2nd_grad=False),
    'lrelu':    EasyDict(func=lambda x, alpha, **_:  torch.nn.functional.leaky_relu(x, alpha),   def_alpha=0.2,  def_gain=np.sqrt(2),    cuda_idx=3, ref='y', has_2nd_grad=False),
    'tanh':     EasyDict(func=lambda x, **_:         torch.tanh(x),                              def_alpha=0,    def_gain=1,             cuda_idx=4, ref='y', has_2nd_grad=True),
    'sigmoid':  EasyDict(func=lambda x, **_:         torch.sigmoid(x),                           def_alpha=0,    def_gain=1,             cuda_idx=5, ref='y', has_2nd_grad=True),
    'elu':      EasyDict(func=lambda x, **_:         torch.nn.functional.elu(x),                 def_alpha=0,    def_gain=1,             cuda_idx=6, ref='y', has_2nd_grad=True),
    'selu':     EasyDict(func=lambda x, **_:         torch.nn.functional.selu(x),                def_alpha=0,    def_gain=1,             cuda_idx=7, ref='y', has_2nd_grad=True),
    'softplus': EasyDict(func=lambda x, **_:         torch.nn.functional.softplus(x),            def_alpha=0,    def_gain=1,             cuda_idx=8, ref='y', has_2nd_grad=True),
    'swish':    EasyDict(func=lambda x, **_:         torch.sigmoid(x) * x,                       def_alpha=0,    def_gain=np.sqrt(2),    cuda_idx=9, ref='x', has_2nd_grad=True),
}

#----------------------------------------------------------------------------

_inited = False
_plugin = None
_null_tensor = torch.empty([0])


def bias_act(x, b=None, dim=1, act='linear', alpha=None, gain=None, clamp=None, impl='cuda'):
    r"""Fused bias and activation function.

    Adds bias `b` to activation tensor `x`, evaluates activation function `act`,
    and scales the result by `gain`. Each of the steps is optional. In most cases,
    the fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports first and second order gradients,
    but not third order gradients.

    Args:
        x:      Input activation tensor. Can be of any shape.
        b:      Bias vector, or `None` to disable. Must be a 1D tensor of the same type
                as `x`. The shape must be known, and it must match the dimension of `x`
                corresponding to `dim`.
        dim:    The dimension in `x` corresponding to the elements of `b`.
                The value of `dim` is ignored if `b` is not specified.
        act:    Name of the activation function to evaluate, or `"linear"` to disable.
                Can be e.g. `"relu"`, `"lrelu"`, `"tanh"`, `"sigmoid"`, `"swish"`, etc.
                See `activation_funcs` for a full list. `None` is not allowed.
        alpha:  Shape parameter for the activation function, or `None` to use the default.
        gain:   Scaling factor for the output tensor, or `None` to use default.
                See `activation_funcs` for the default scaling of each activation function.
                If unsure, consider specifying 1.
        clamp:  Clamp the output values to `[-clamp, +clamp]`, or `None` to disable
                the clamping (default).
        impl:   Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

    Returns:
        Tensor of the same shape and datatype as `x`.
    """
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'cuda']
    return _bias_act_ref(x=x, b=b, dim=dim, act=act, alpha=alpha, gain=gain, clamp=clamp)

# ----------------------------------------------------------------------------


def _bias_act_ref(x, b=None, dim=1, act='linear', alpha=None, gain=None, clamp=None):
    """Slow reference implementation of `bias_act()` using standard TensorFlow ops.
    """
    assert isinstance(x, torch.Tensor)
    assert clamp is None or clamp >= 0
    spec = activation_funcs[act]
    alpha = float(alpha if alpha is not None else spec.def_alpha)
    gain = float(gain if gain is not None else spec.def_gain)
    clamp = float(clamp if clamp is not None else -1)

    # Add bias.
    if b is not None:
        assert isinstance(b, torch.Tensor) and b.ndim == 1
        assert 0 <= dim < x.ndim
        assert b.shape[0] == x.shape[dim]
        x = x + b.reshape([-1 if i == dim else 1 for i in range(x.ndim)])

    # Evaluate activation function.
    alpha = float(alpha)
    x = spec.func(x, alpha=alpha)

    # Scale by gain.
    gain = float(gain)
    if gain != 1:
        x = x * gain

    # Clamp.
    if clamp >= 0:
        x = x.clamp(-clamp, clamp) # pylint: disable=invalid-unary-operand-type
    return x

