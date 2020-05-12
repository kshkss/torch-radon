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
        assert sino.shape == (angles.shape[0], sino_width)

        sizes = torch.tensor([width, height, sino_width], dtype=torch.int)
        centers = torch.tensor([x_center, y_center, sino_center], dtype=torch.float32)
        ctx.save_for_backward(angles, sizes, centers)

        return sino

    @staticmethod
    def backward(ctx, grad_sino):
        angles, sizes, centers = ctx.saved_tensors
        width, height, sino_width = sizes[0], sizes[1], sizes[2]
        x_center, y_center, sino_center = centers[0], centers[1], centers[2]

        assert grad_sino.shape == (angles.shape[0], sino_width)
        if ctx.needs_input_grad[0]:
            grad_tomo = radon_cpp.backward(grad_sino, angles, width, height, x_center, y_center, sino_center)
            assert grad_tomo.shape == (height, width)
        else:
            grad_tomo = None

        return grad_tomo, None, None, None, None, None

radon = RadonFunction.apply

class BackprojectionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sino, angles, width=None, height=None, x_center=None, y_center=None, sino_center=None):
        assert sino.ndim == 2
        assert angles.ndim == 1
        assert angles.shape[0] == sino.shape[0]

        sino_width = sino.shape[1]
        if width is None:
            width = sino_width
        if height is None:
            height = sino_width
        if x_center is None:
            x_center = 0.5 * (width - 1)
        if y_center is None:
            y_center = 0.5 * (height - 1)
        if sino_center is None:
            sino_center = 0.5 * (sino_width - 1)

        tomo = radon_cpp.backward(sino, angles, width, height, x_center, y_center, sino_center)
        assert tomo.shape == (height, width)

        sizes = torch.tensor([width, height, sino_width], dtype=torch.int)
        centers = torch.tensor([x_center, y_center, sino_center], dtype=torch.float32)
        ctx.save_for_backward(angles, sizes, centers)

        return tomo

    @staticmethod
    def backward(ctx, grad_tomo):
        angles, sizes, centers = ctx.saved_tensors
        width, height, sino_width = sizes[0], sizes[1], sizes[2]
        x_center, y_center, sino_center = centers[0], centers[1], centers[2]

        assert grad_tomo.shape == (height, width)
        if ctx.needs_input_grad[0]:
            grad_sino = radon_cpp.forward(grad_tomo, angles, sino_width, x_center, y_center, sino_center)
            assert grad_sino.shape == (angles.shape[0], sino_width)
        else:
            grad_sino = None

        return grad_sino, None, None, None, None, None, None

backprojection = BackprojectionFunction.apply

