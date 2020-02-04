from autograd import *

# Declare variables
x = Variable('x')

# Define function f(x) = 4x^4 + 5x^3 + x^2 + 18
f1 = C(4)*x**C(4) + C(5)*x**C(3) + x**C(2) + C(18)
print('f1:', f1)

# Compute the gradient (this returns the computation graphs of each component of the gradient function)
g1 = f1.gradient()
print('g1:', g1)

# Differentiate in which point?
in_point = {x: 3}

# Evaluate the `x` component of the gradient at `in_point`
print('x component of gradient:', g1[x].forward(in_point))

# Verify with numerical gradient
print('numerical gradient:', f1.numerical_diff_grad(in_point))

# We can also just evaluate f1 at `in_point` (the forward pass)
print('f1 evaluated:', f1.forward(in_point), '\n\n')


x = Variable('x')
y = Variable('y')
z = Variable('z')

f2 = Ln(x/y) + x**y + z**C(-3) + Exp(y/x)
print('f2:', f2)
in_point = {x: 6, y: 3, z: 5}
for v, k in f2.gradient().items():
    print(f'\tf2 {v}-component of gradient: {k}')
    print('\tevaluated at in_point: ', k.forward(in_point))
print('numerical gradient:', f2.numerical_diff_grad(in_point, h=0.0001))
