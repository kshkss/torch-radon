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
# Import torch followed by torch_radon.
import torch
from torch_radon import radon, backprojection
import math

# Define projection angles for radon transform.
angles = math.pi * torch.arange(200, dtype=torch.float32) / 200.0

# Load an image file as a Tensor that has 32-bit float elements and is allocated on gpu.
image = torch.randn((256, 256), dtype=torch.float32, device='cuda', requires_grad=True)

# Call radon transform
radon_results = radon(image, angles)

# 
sinogram = torch.randn((200, 256), dtype=torch.float32, device='cuda')
inner_product = (sinogram * radon_results).sum()
inner_product.backward()
image.grad    ##= backprojection(sinogram, angles)

# Evaluation for the gradient is computed indirectly and include some error.
error = (sinogram * radon_results).sum() - (backprojection(sinogram, angles) * image).sum()
```

