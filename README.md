# torch-radon
A PyTorch extension to compute radon transform and its adjoint operator using gpu

## Dependency
* PyTorch 
* CUDA toolkit

## Install
Pip is available to install modules.
```
pip install git+https://github.com/kshkss/torch-radon
```

## Usage
```
import torch
from torch_radon import radon, backprojection
import math

n_angles = 200
width = 256

# Define projection angles for radon transform.
angles = math.pi * torch.arange(n_angles, dtype=torch.float32) / n_angles

# Load an image file as a Tensor that has 32-bit float elements and is allocated on gpu.
image = torch.randn((width, width), dtype=torch.float32, device='cuda', requires_grad=True)

# Call radon transform
radon_results = radon(image, angles)

#
sinogram = torch.randn((n_angles, width), dtype=torch.float32, device='cuda')
inner_product = (sinogram * radon_results).sum()
inner_product.backward()
image.grad    ##= backprojection(sinogram, angles)

# Evaluation for the gradient is computed indirectly and include some error.
error = (sinogram * radon(image, angles)).sum() - (backprojection(sinogram, angles) * image).sum()
print(error)
```

