import autograd.numpy as anp
from autograd import grad

x = anp.array([2.]).astype(anp.float32)
def f(x):
    return 3*x+2
fprime = grad(f)
print(f(x), '---', fprime(x))
