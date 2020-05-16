import math
import torch
from logging import getLogger

logger = getLogger(__name__)

assert (torch.cuda.is_available())
import radon_cuda as radon_cpp

class RadonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tomo, angles, sino_width=None, x_center=None, y_center=None, sino_center=None, out=None):
        assert tomo.ndim == 2
        assert angles.ndim == 1

        height = tomo.shape[0]
        width = tomo.shape[1]
        
        if out is None:
            if sino_width is None:
                sino_width = width
            out = torch.empty((angles.shape[0], sino_width), dtype=torch.float64, device=tomo.device)
        else:
            assert(angles.shape[0] == out.shape[0])
            if sino_width is None:
                sino_width = out.shape[1]
            else:
                assert(sino_width == out.shape[1])

        if x_center is None:
            x_center = 0.5 * (width - 1)
        if y_center is None:
            y_center = 0.5 * (height - 1)
        if sino_center is None:
            sino_center = 0.5 * (sino_width - 1)

        sino = radon_cpp.forward(out, tomo, angles, x_center, y_center, sino_center)

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
            grad_tomo = torch.empty((height, width), dtype=torch.float32, device=grad_sino.device)
            if grad_sino.dtype == torch.float64:
                logger.debug('Backward of radon: the precision of input gradients is fallback to FP32')
            radon_cpp.backward(grad_tomo, grad_sino.to(torch.float32), angles, x_center, y_center, sino_center)
        else:
            grad_tomo = None

        return grad_tomo, None, None, None, None, None

radon = RadonFunction.apply

class BackprojectionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sino, angles, width=None, height=None, x_center=None, y_center=None, sino_center=None, out=None):
        assert sino.ndim == 2
        assert angles.ndim == 1
        assert angles.shape[0] == sino.shape[0]

        sino_width = sino.shape[1]
        if out is None:
            if width is None:
                width = sino_width
            if height is None:
                height = sino_width
            out = torch.empty((height, width), dtype=torch.float64, device=sino.device)
        else:
            if width is None:
                width = out.shape[1]
            else:
                assert(width == out.shape[1])
            if height is None:
                height = out.shape[0]
            else:
                assert(height == out.shape[0])

        if x_center is None:
            x_center = 0.5 * (width - 1)
        if y_center is None:
            y_center = 0.5 * (height - 1)
        if sino_center is None:
            sino_center = 0.5 * (sino_width - 1)

        tomo = radon_cpp.backward(out, sino, angles, x_center, y_center, sino_center)
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
            grad_sino = torch.empty((angles.shape[0], sino_width), dtype=torch.float32, device=grad_tomo.device)
            if grad_tomo.dtype == torch.float64:
                logger.debug('Backward of backprojection: the precision of input gradients is fallback to FP32')
            radon_cpp.forward(grad_sino, grad_tomo.to(torch.float32), angles, x_center, y_center, sino_center)
        else:
            grad_sino = None

        return grad_sino, None, None, None, None, None, None

backprojection = BackprojectionFunction.apply

