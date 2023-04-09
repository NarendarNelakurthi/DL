from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,wrap_tuple,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """
    `ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes:
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    def __init__(self, last_fn=None, ctx=None, inputs=None):
        self.last_fn = last_fn
        self.ctx = ctx
        self.inputs = inputs


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


class Scalar:
    """
    A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """
    def __init__(self, v, back=ScalarHistory(), name=None):
        super().__init__(back, name=name)
        self.data = float(v)
    def __repr__(self):
        return "Scalar(%f)" % self.data
    
    def __mul__(self, b):
        return Mul.apply(self, b)
    
    def __truediv__(self, b):
        return Mul.apply(self, Inv.apply(b))
    
    def __rtruediv__(self, b):
        return Mul.apply(b, Inv.apply(self))

    def __add__(self, b):
        # TODO: Implement for Task 1.2. DONE
        # raise NotImplementedError('Need to implement for Task 1.2')
        return Add.apply(self, b)

    def __bool__(self):
        return bool(self.data)

    def __lt__(self, b):
        # TODO: Implement for Task 1.2. DONE
        # raise NotImplementedError('Need to implement for Task 1.2')
        return LT.apply(self, b)

    def __gt__(self, b):
        # TODO: Implement for Task 1.2. DONE
        # raise NotImplementedError('Need to implement for Task 1.2')
        return LT.apply(-self, -b)

    def __eq__(self, b):
        # TODO: Implement for Task 1.2. DONE
        # raise NotImplementedError('Need to implement for Task 1.2')
        return EQ.apply(self, b)

    def __sub__(self, b):
        # TODO: Implement for Task 1.2. DONE
        # raise NotImplementedError('Need to implement for Task 1.2')
        return Add.apply(self, -b)

    def __neg__(self):
        # TODO: Implement for Task 1.2. DONE
        # raise NotImplementedError('Need to implement for Task 1.2')
        return Neg.apply(self)

    def log(self):
        # TODO: Implement for Task 1.2. DONE
        # raise NotImplementedError('Need to implement for Task 1.2')
        return Log.apply(self)

    def exp(self):
        # TODO: Implement for Task 1.2. DONE
        # raise NotImplementedError('Need to implement for Task 1.2')
        return Exp.apply(self)

    def sigmoid(self):
        # TODO: Implement for Task 1.2. DONE
        # raise NotImplementedError('Need to implement for Task 1.2')
        return Sigmoid.apply(self)

    def relu(self):
        # TODO: Implement for Task 1.2. DONE
        # raise NotImplementedError('Need to implement for Task 1.2')
        return ReLU.apply(self)

    def get_data(self):
        "Returns the raw float value"
        return self.data

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """
        Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
            x: value to be accumulated
        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.derivative = 0.0
        self.derivative += x

    def is_leaf(self) -> bool:
        "True if this variable created by the user (no `last_fn`)"
        return self.history.last_fn is None

    def is_constant(self):
        return not isinstance(self, Variable) or self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, inputs, d_output) -> Iterable[Tuple[Variable, Any]]:
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        # TODO: Implement for Task 1.3.
        # raise NotImplementedError("Need to implement for Task 1.3")
        derivatives = wrap_tuple(Scalar.backward(self, d_output))
        return [
            (inp, deriv)
            for inp, deriv in zip(inputs, derivatives)
            if not Scalar.is_constant(inp)
        ]

    def backward(self, d_output: Optional[float] = None) -> None:
        """
        Calls autodiff to fill in the derivatives for the history of this object.

        Args:
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).
        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """
    Checks that autodiff works on a python function.
    Asserts False if derivative is incorrect.

    Parameters:
        f : function from n-scalars to 1-scalar.
        *scalars  : n input scalar values.
    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
