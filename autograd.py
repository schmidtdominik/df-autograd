import itertools
import math
import string
from abc import abstractmethod, ABCMeta
import random
from typing import Dict, Type, Set

"""

This is a simple autodifferentiation experiment
`Node` computes first order derivatives with respect to all variables in the graph.
I also implemented numeric differentiation to verify my results.

Instead of a backward() method each Node has a gradient() method which returns the gradient's computation graph.
This allows for easy computation of higher order derivatives.

"""

class Node(metaclass=ABCMeta):

    @abstractmethod
    def forward(self, variable_assignments: Dict['Variable', int]):
        pass

    @abstractmethod
    def gradient(self, upstream_gradient: Dict['Variable', 'Node'] = None):
        pass

    def simplify(self):
        return self

    def __pow__(self, other):
        return Pow(self, other)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Mult(self, other)

    def __truediv__(self, other):
        return Div(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __rmul__(self, other):
        return Mult(other, self)

    def __rtruediv__(self, other):
        return Div(other, self)

    def __rsub__(self, other):
        return Sub(other, self)

    @property
    def variables(self) -> Set:
        pass

    def numerical_diff_grad(self, in_point, h=0.000001):
        # in_point: in which point to numerically differentiate
        assert len(set(self.variables) - set(in_point.keys())) == 0

        gradient = {}
        for v in self.variables:
            a = in_point.copy()
            a[v] += h
            b = in_point.copy()
            b[v] -= h
            gradient[v] = (self.forward(a)-self.forward(b))/(2*h)
        return gradient

    @staticmethod
    def merge_grads(grad1: Dict, grad2: Dict, op: Type['BinaryOp']):
        result = {}
        for var in itertools.chain(grad1.keys(), grad2.keys()):
            if var in grad1 and var in grad2:
                result[var] = Add(grad1[var], grad2[var])
            elif var in grad1:
                result[var] = grad1[var]
            else:
                result[var] = grad2[var]
        return result

    @staticmethod
    def broadcast_mult(gradient, value):
        return {key: value.simplify() if (type(gradient[key]) is C and gradient[key].value == 1) else (gradient[key] * value).simplify() for key in gradient}

class C(Node):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f'{self.value}'

    def forward(self, variable_assignments):
        return self.value

    def gradient(self, upstream_gradient: Dict['Variable', 'Node'] = None):
        return {}

    @property
    def variables(self):
        return set()


class Variable(Node):
    def __init__(self, name):
        # variables that have the same name are treated as equal
        self.name = name

    def __repr__(self):
        return f'{self.name}'

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def forward(self, variable_assignments):
        """
        forward() can raise the following exceptions:
            - ValueError (math domain error)
            - OverflowError (numerical result out of range/complex exponentiation)
            - TypeError (can't convert complex to float)
            - ZeroDivisionError
        """
        return variable_assignments[self]

    def gradient(self, upstream_gradient = None):
        return self.broadcast_mult({self: C(1)}, upstream_gradient) if upstream_gradient else {self: C(1)}

    def simplify(self):
        return self

    @property
    def variables(self):
        return {self}


class BinaryOp(Node):
    symbol = None

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __repr__(self):
        return f'({repr(self.a)} {self.symbol} {repr(self.b)})'

    @property
    def variables(self):
        return self.a.variables | self.b.variables

    def simplify(self):
        if type(self.a) is C and type(self.b) is C:
            return C(self.forward({}))

        self.a = self.a.simplify()
        self.b = self.b.simplify()
        return self

class UnaryOp(Node):
    symbol = None

    def __init__(self, a):
        self.a = a

    def __repr__(self):
        return f'{self.symbol} {repr(self.a)}'

    @property
    def variables(self):
        return self.a.variables

    def simplify(self):
        self.a = self.a.simplify()
        return self


class Add(BinaryOp):
    symbol = '+'

    def forward(self, variable_assignments):
        return self.a.forward(variable_assignments) + self.b.forward(variable_assignments)

    def gradient(self, upstream_gradient=None):
        local_grad = self.merge_grads(self.a.gradient(), self.b.gradient(), op=Add)
        return self.broadcast_mult(local_grad, upstream_gradient) if upstream_gradient else local_grad

class Sub(BinaryOp):
    symbol = '-'

    def forward(self, variable_assignments):
        return self.a.forward(variable_assignments) - self.b.forward(variable_assignments)

    def gradient(self, upstream_gradient=None):
        local_grad = self.merge_grads(self.a.gradient(), Mult(C(-1), self.b).gradient(), op=Add)
        return self.broadcast_mult(local_grad, upstream_gradient) if upstream_gradient else local_grad

class Mult(BinaryOp):
    symbol = '*'

    def forward(self, variable_assignments):
        return self.a.forward(variable_assignments) * self.b.forward(variable_assignments)

    def gradient(self, upstream_gradient=None):
        local_grad = self.merge_grads(self.a.gradient(upstream_gradient=self.b), self.b.gradient(upstream_gradient=self.a), op=Add)
        return self.broadcast_mult(local_grad, upstream_gradient) if upstream_gradient else local_grad

    def simplify(self):
        self.a = self.a.simplify()
        self.b = self.b.simplify()

        if type(self.a) is Mult and type(self.b) is C:
            if type(self.a.a) is C:
                return Mult(C((self.b*self.a.a).forward({})), self.a.b)
            elif type(self.a.b) is C:
                return Mult(C((self.b*self.a.b).forward({})), self.a.a)
        elif type(self.b) is Mult and type(self.a) is C:
            if type(self.b.a) is C:
                return Mult(C((self.a*self.b.a).forward({})), self.b.b)
            elif type(self.b.b) is C:
                return Mult(C((self.a*self.b.b).forward({})), self.b.a)

        return super().simplify()


class Div(BinaryOp):
    symbol = '/'

    def forward(self, variable_assignments):
        return self.a.forward(variable_assignments) / self.b.forward(variable_assignments)

    def gradient(self, upstream_gradient=None):
        return (self.a * self.b ** C(-1)).gradient(upstream_gradient=upstream_gradient)

class Ln(UnaryOp):
    symbol = 'ln'

    def forward(self, variable_assignments):
        return math.log(self.a.forward(variable_assignments))

    def gradient(self, upstream_gradient=None):
        local_grad = self.a.gradient(upstream_gradient=Div(C(1), self.a))
        return self.broadcast_mult(local_grad, upstream_gradient) if upstream_gradient else local_grad

class Exp(UnaryOp):
    symbol = 'exp'

    def forward(self, variable_assignments):
        return math.exp(self.a.forward(variable_assignments))

    def gradient(self, upstream_gradient=None):
        local_grad = self.a.gradient(upstream_gradient=Exp(self.a))
        return self.broadcast_mult(local_grad, upstream_gradient) if upstream_gradient else local_grad


class Pow(BinaryOp):
    symbol = '^'

    def forward(self, variable_assignments):
        return self.a.forward(variable_assignments) ** self.b.forward(variable_assignments)

    def gradient(self, upstream_gradient=None):
        """
        f(u, v) = u^v
        v*u^(v-1)  * u`  + u^v * ln(u) * v`
        """
        local_grad = self.merge_grads(self.a.gradient(upstream_gradient=self.b*self.a**(self.b - C(1))), self.b.gradient(upstream_gradient=self.a ** self.b * Ln(self.a)), op=Add)
        return self.broadcast_mult(local_grad, upstream_gradient) if upstream_gradient else local_grad

def randomized_eq(depth=1):
    binary_ops = [Add, Mult, Sub, Div, Pow]
    unary_ops = [Ln, Exp]

    k = random.random()*math.exp(-depth/10)

    if k < 1/7:
        return Variable(random.choice(string.ascii_lowercase[:7]))
    elif k < 2/7:
        return C(random.randint(-10, 10))
    else:
        op = random.choice(binary_ops + unary_ops)
        if op in binary_ops:
            return op(randomized_eq(depth+1), randomized_eq(depth+1))
        else:
            return op(randomized_eq(depth+1))