#include <torch/extension.h>
#include <cassert>

// CUDA declarations

torch::Tensor radon_cuda_forward(
	torch::Tensor tomo,
	torch::Tensor angles,
	int width_sino,
	float x_center,
	float y_center,
	float u_center);

torch::Tensor radon_cuda_backward(
	torch::Tensor sino,
	torch::Tensor angles,
	int width,
	int height,
	float x_center,
	float y_center,
	float u_center);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor radon_forward(
    torch::Tensor tomo,
    torch::Tensor angles,
    int width_sino,
    float x_center,
    float y_center,
    float u_center)
{
	  CHECK_INPUT(tomo);

	  torch::Tensor angles_;
	  if( angles.device().is_cuda()
		  && angles.device().index() == tomo.device().index()
		  && angles.dtype() == torch::kFloat32 ) 
	  {
		  angles_ = angles;
	  }else{
		  angles_ = angles.to(torch::dtype(torch::kFloat32).device(torch::kCUDA, tomo.device().index()));
	  }

	  return radon_cuda_forward(tomo, angles_, width_sino, x_center, y_center, u_center);
}

torch::Tensor radon_backward(
    torch::Tensor sino,
    torch::Tensor angles,
    int width,
    int height,
    float x_center,
    float y_center,
    float u_center)
{
	  CHECK_INPUT(sino);

	  torch::Tensor angles_;
	  if( angles.device().is_cuda()
		  && angles.device().index() == sino.device().index()
		  && angles.dtype() == torch::kFloat32 ) 
	  {
		  angles_ = angles;
	  }else{
		  angles_ = angles.to(torch::dtype(torch::kFloat32).device(torch::kCUDA, sino.device().index()));
	  }

	  return radon_cuda_backward(sino, angles_, width, height, x_center, y_center, u_center);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &radon_forward, "Radon transform forward (CUDA)");
  m.def("backward", &radon_backward, "Radon transform backward (CUDA)");
}

