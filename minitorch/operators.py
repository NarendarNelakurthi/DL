"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    "$f(x, y) = x * y$"
    return x * y


def id(x):
    """Identity function

    Args:
        x : any input

    Returns:
        x : f(x) = x
    """
    return x


def add(x, y):
    """Adds two scalars

    Args:
        x : scalar input x
        y : scalar input y

    Returns:
        f(x, y) = x + y
    """
    return x + y


add


def neg(x):
    """Negates the input

    Args:
        x: the input to negate

    Returns:
        f(x) = -x
    """
    return -x


neg


def lt(x, y):
    """less than function

    Args:
        x : the input value
        y : the value to compare against

    Returns:
        1.0 if x is less than y else 0.0
    """

    if x < y:
        return 1.0
    else:
        return 0.0


def eq(x, y):
    """test equality function

    Args:
        x : the input value
        y : the value to compare against

    Returns:
        1.0 if x is equal to y else 0.0
    """

    if x == y:
        return 1.0
    else:
        return 0.0


def max(x, y):
    """maximum value function

    Args:
        x : first input
        y : second input

    Returns:
        x if x is greater than y else y
    """

    if x > y:
        return x
    else:
        return y


def is_close(x, y):
    """Returns true if two values are close within
    a threshold value

    Args:
        x : value 1
        y : value 2

    Returns:
        f(x) = |x - y| < 1e-2
    """

    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """sigmoid activation function
    (See `<https://en.wikipedia.org/wiki/Sigmoid_function>`_ .)

    Calculated as

    f(x) =  1/(1.0 + e^{-x}) if x >=0 else e^x/(1.0 + e^{x})

    for stability.

    Args:
        x (float): input value

    Returns:
        float : sigmoid value
    """
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """ReLu activation function

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

    Args:
        x (float): input

    Returns:
        float : relu value
    """
    if x > 0:
        return x
    else:
        return 0.0


EPS = 1e-6


def log(x):
    """Return natural log with small value (e) added

    Args:
        x ([type]): input

    Returns:
        f(x) = log(x + e)
    """

    return math.log(x + EPS)


def exp(x):
    """exponential function

    Args:
        x : input value

    Returns:
        f(x) = e^(x)
    """

    return math.exp(x)


def log_back(x, d):
    """returns d * f'(x) where f = log(x)

    Args:
        x : activation value
        d : derivative

    Returns:
        d * f'(x) where f = log(x)
    """

    return d * (1 / x)


def inv(x):
    """inverse function

    Args:
        x : input value

    Returns:
        f(x) = 1/x
    """

    return 1 / x


def inv_back(x, d):
    """returns d * f'(x) where f = 1/x

    Args:
        x : activation value
        d : derivative

    Returns:
        d * f'(x) where f = 1/x
    """

    return d * -1 / (x ** 2)


def relu_back(x, d):
    """returns d * f'(x) where f = relu(x)

    Args:
        x : activation value
        d : derivative

    Returns:
        d * f'(x) where f = relu(x)
    """

    if x > 0:
        return 1 * d
    else:
        return 0


# ## Task 0.3

# Small library of elementary higher-order functions for practice.


def map(fn):
    """
    Higher-order map.

    .. image:: figs/Ops/maplist.png


    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (one-arg function): Function from one value to one value.

    Returns:
        function : A function that takes a list, applies `fn` to each element, and returns a
        new list
    """
    return lambda l: [fn(i) for i in l]


def negList(ls):
    """negates all elements in input list

    Args:
        ls (list): the list to be negated

    Returns:
        list : a list with the neg(x) function mapped to the values
    """
    mapper = map(neg)

    return mapper(ls)


def zipWith(fn):
    """
    Higher-order zipwith (or map2).

    .. image:: figs/Ops/ziplist.png

    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (two-arg function): combine two values

    Returns:
        function : takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) on each pair of elements.

    """

    return lambda ls1, ls2: [fn(x, y) for x, y in zip(ls1, ls2)]


def addLists(ls1, ls2):
    """the elementwise addition of two list

    Args:
        ls1 (list): the first list
        ls2 (list): the second list

    Returns:
        list: the elementwise addition of ls and ls2

    """

    zip = zipWith(add)
    return zip(ls1, ls2)


def reduce(fn, start):
    """
    Higher-order reduce.

    Args:
        fn (two-arg function): combine two values
        start (float): start value :math:`x_0`

    Returns:
        function : function that takes a list `ls` of elements
        and computes the reduction fn(x_3,fn(x_2, fn(x_1, x_0)))

    """

    def reduced(ls):
        r = start
        for el in ls:
            r = fn(r, el)
        return r

    return reduced


def sum(ls):
    """the cumulative sum of all elements in a list

    Args:
        ls (list): the list to sum

    Returns:
        scalar: the cumulative sum of ls
    """

    summ = reduce(add, 0)
    return summ(ls)


def prod(ls):
    """the cumulative product of all elements in a list

    Args:
        ls (list): the list to find cum product of

    Returns:
        scalar: the cumulative product of ls
    """

    producer = reduce(mul, 1)
    return producer(ls)