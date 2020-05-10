import math
import torch

assert (torch.cuda.is_available())
import radon_cuda as radon_cpp

class RadonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tomo, angles, sino_width=None, x_center=None, y_center=None, sino_center=None):
        assert tomo.ndim == 2
        assert angles.ndim == 1

        height = tomo.shape[0]
        width = tomo.shape[1]
        if sino_width is None:
            sino_width = width
        if x_center is None:
            x_center = 0.5 * (width - 1)
        if y_center is None:
            y_center = 0.5 * (height - 1)
        if sino_center is None:
            sino_center = 0.5 * (sino_width - 1)

        sino = radon_cpp.forward(tomo, angles, sino_width, x_center, y_center, sino_center)
        ctx.save_for_backward(angles, width, height, sino_width, x_center, y_center, sino_center)

        return sino

    @staticmethod
    def backward(ctx, grad_sino):
        angles, width, height, sino_width, x_center, y_center, sino_center = ctx.saved_tensors
        assert grad_sino.shape == (angles.shape[0], sino_width)

        if ctx.needs_input_grad[0]:
            grad_tomo = radon_cpp.backward(grad_sino, angles, width, height, x_center, y_center, sino_center)
        else:
            grad_tomo = None

        return grad_tomo, None, None, None, None, None

radon = RadonFunction.apply

