#include <cassert>
#include <cstdint>

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __inline__
static void compute_backprojection(float *acc,
		cudaTextureObject_t sino, int umax, int t,
		float uc, float theta,
		float x, float y)
{
	float u = uc + cos(theta) * x - sin(theta) * y;
	if(u >= 0 && u < umax){
		*acc += tex2D<float>(sino, u, (float)t);
	}
}

__global__
static void backprojection_gpu_calc(float *tomo, cudaTextureObject_t sino,
		int width, int height, int umax,
		int n_angles, float *angles,
		float xc, float yc, float uc)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = blockIdx.y;
	
	float acc = 0.0;
	for(int i = 0; i < n_angles; i++){
		compute_backprojection(&acc, sino, umax, i, uc, angles[i], (float)x - xc, (float)y - yc);
	}

	if(x < width){
		tomo[width * y + x] = acc;
	}
}

constexpr int n_threads = 256;

void backprojection_gpu(float *tomo, const float *sino,
		int width, int height, int umax,
		int n_angles, float *angles,
		float xc, float yc, float uc)
{
	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray *sino_;
	cudaMallocArray(&sino_, &channelDesc, umax, n_angles);
	cudaMemcpy2DToArray(sino_, 0, 0, sino, umax*sizeof(float), umax*sizeof(float), n_angles, cudaMemcpyHostToDevice);

	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = sino_;

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = false;

	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	dim3 block(n_threads);
	dim3 grid((width + block.x - 1) / block.x, height, 1);
	backprojection_gpu_calc<<<grid, block>>>(tomo, texObj, width, height, umax, n_angles, angles, xc, yc, uc);

	cudaDestroyTextureObject(texObj);
	cudaFreeArray(sino_);
}

torch::Tensor radon_cuda_backward(
		torch::Tensor sino,
		torch::Tensor angles,
		int width,
		int height,
		float x_center,
		float y_center,
		float u_center)
{
	AT_DISPATCH_FLOATING_TYPES(sino.type(), "radon_cuda_backward", ([&] {
		if(sizeof(scalar_t) != 32){
			AT_ERROR("radon_cuda_backward is implemented for only 32-bit floating point");
		}else{
			int n_angles = sino.size(0);
			int width_sino = sino.size(1);

			auto options = torch::TensorOptions()
				.dtype(torch::kFloat32)
				.device(torch::kCUDA, sino.device().index());
			torch::Tensor tomo = torch::empty({height, width}, options);

			backprojection_gpu(
				tomo.data_ptr<float>(),
				sino.data_ptr<float>(),
				width,
				height,
				width_sino,
				n_angles,
				angles.data_ptr<float>(),
				x_center,
				y_center,
				u_center);
			return sino;
		}
	}));
}

