from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
from collections import defaultdict
from scalar import ScalarHistory
from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    # raise NotImplementedError("Need to implement for Task 1.1")
    x = list(vals)
    x[arg] -= epsilon
    y = list(vals)
    y[arg] += epsilon
    out = (f(*y) - f(*x)) / (2 * epsilon)
    return out


variable_count = 1


class Variable:
    
    def __init__(self, history, name=None):
        global variable_count
        assert history is None or isinstance(history, ScalarHistory), history

        self.history = history
        self._derivative = None

        # This is a bit simplistic, but make things easier.
        variable_count += 1
        self.unique_id = "Variable" + str(variable_count)

        # For debugging can have a name.
        if name is not None:
            self.name = name
        else:
            self.name = self.unique_id
        self.used = 0


    @property
    def derivative(self):
        return self._derivative

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        return self.history.last_fn is None

    def is_constant(self) -> bool:
        return not isinstance(self, Variable) or self.history is None
    
    def accumulate_derivative(self, val):
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self._derivative is None:
            self._derivative = self.zeros()
        self._derivative += val
    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(node) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError("Need to implement for Task 1.4")
    sorted_nodes = []
    perm_visited = {}
    temp_visited = {}

    def visit(node):

        if not isinstance(node, Variable):
            return

        if node.name in perm_visited:
            return

        if node.name in temp_visited:
            raise Exception(f"{node} is not a Dag")

        temp_visited[node.name] = node

        try:
            for child_node in node.history.inputs:
                visit(child_node)
        except (TypeError, AttributeError):
            pass

        del temp_visited[node.name]
        perm_visited[node.name] = node
        sorted_nodes.append(node)

    visit(node)

    return sorted_nodes[::-1]



def backpropagate(variable, deriv) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError("Need to implement for Task 1.4")
    sorted_nodes = topological_sort(variable)
    accumulated_derivatives = defaultdict(lambda: 0)
    accumulated_derivatives[sorted_nodes[0].name] += deriv

    for node in sorted_nodes:
        try:
            if node.is_leaf():
                node.accumulate_derivative(accumulated_derivatives[node.name])
            else:
                local_derivatives_wrt_inputs = node.history.backprop_step(
                    accumulated_derivatives[node.name]
                )
                for sub_node, local_deriv in local_derivatives_wrt_inputs:
                    accumulated_derivatives[sub_node.name] += local_deriv

        except AttributeError:
            if node._derivative is None:
                node._derivative = 0.0
            node._derivative += accumulated_derivatives[node.name]


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
