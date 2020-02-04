from autograd import *
import numpy as np

deviation = []
for i in range(1000):
    f = randomized_eq()
    printed = False
    try:
        g = f.gradient()
        vars = {v: (random.random() - 0.5) * 30 for v in f.variables}
        num_g = f.numerical_diff_grad(vars)
        print('f:', f)
        printed = True
        if not g:
            print('→ no gradient to compute')

        for v, k in g.items():
            print(f'\t{v}-component of gradient: {k}')
            val = k.forward(vars)
            print('\t.. evaluated at `vars`: ', val)
            print('\t.. numerical grad: ', num_g[v])
            deviation.append(abs(val-num_g[v]))
            if abs(val-num_g[v]) > 1e-2:
                print(f'\t[Warning]: abs(val-num_g[v]) == {abs(val-num_g[v])} > 1e-2')
    except (ValueError, OverflowError, TypeError, ZeroDivisionError) as e:
        # we don't care about math exceptions for div by zero, complex exponentiation, casting complex to float, overflows,..
        if printed:
            print('→', e)
    if printed:
        print()

print('median deviation exact vs. numerical gradient =', np.median(deviation))
