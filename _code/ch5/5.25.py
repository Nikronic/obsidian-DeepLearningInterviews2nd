import numpy as np
import torch
from torch.autograd import Function
from torch.autograd import gradcheck
from torch.autograd import grad

from typing import Any


class ArcTanh(Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        x = x.detach().numpy()
        z = np.arctanh(x)
        x = torch.from_numpy(x)
        z = torch.from_numpy(z)
        ctx.save_for_backward(x)
        return z

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors
        if isinstance(x, tuple):
            if len(x) == 1:
                x = x[0]
                grad_outputs = grad_outputs[0]
        z = 1./(1. - x**2)
        return z * grad_outputs


x = np.array([0.37, 0.192, 0.571]).astype(np.float32)
x_pt = torch.from_numpy(x).requires_grad_(True)

# np eval
y_np = np.arctanh(x)
y_prime = 1./(1. - x**2)

# pt eval
pt_arctanh = ArcTanh().apply
y_pt = pt_arctanh(x_pt)
y_prime_pt = grad(y_pt, x_pt, only_inputs=True, grad_outputs=torch.ones_like(x_pt), create_graph=True)
test = gradcheck(pt_arctanh, (x_pt,), eps=1e-4, atol=1e-4)
print(test)
