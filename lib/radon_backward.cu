#include <cassert>
#include <cstdint>

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<class scalar_t>
__device__ __inline__
static void compute_backprojection(scalar_t *acc,
		cudaTextureObject_t sino, int umax, int t,
		float uc, float theta,
		float x, float y)
{
	float u = uc + cos(theta) * x - sin(theta) * y;
	if(u >= 0 && u <= umax-1){
		*acc += static_cast<scalar_t>(tex2D<float>(sino, u, (float)t));
	}/*
	}else if(-1 < u && u < 0){
		*acc += (1 + u) * tex2D<float>(sino, 0, (float)t);
	}else if(umax-1 < u && u < umax){
		*acc += (umax - u) * tex2D<float>(sino, umax-1, (float)t);
	}
	*/
}

template<class scalar_t>
__global__
static void backprojection_gpu_calc(scalar_t *tomo, cudaTextureObject_t sino,
		int width, int height, int umax,
		int n_angles, float *angles,
		float xc, float yc, float uc)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = blockIdx.y;
	
	scalar_t acc = 0.0;
	for(int i = 0; i < n_angles; i++){
		compute_backprojection<scalar_t>(&acc, sino, umax, i, uc, angles[i], (float)x - xc, (float)y - yc);
	}

	if(x < width){
		tomo[width * y + x] = acc;
	}
}

constexpr int n_threads = 256;

template<class scalar_t>
void backprojection_gpu(scalar_t *tomo, const float *sino,
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
	backprojection_gpu_calc<scalar_t><<<grid, block>>>(tomo, texObj, width, height, umax, n_angles, angles, xc, yc, uc);

	cudaDestroyTextureObject(texObj);
	cudaFreeArray(sino_);
}

torch::Tensor radon_cuda_backward(
		torch::Tensor tomo,
		torch::Tensor sino,
		torch::Tensor angles,
		float x_center,
		float y_center,
		float u_center)
{
	int n_angles = sino.size(0);
	int width_sino = sino.size(1);
	int height = tomo.size(0);
	int width = tomo.size(1);

	AT_DISPATCH_FLOATING_TYPES(tomo.type(), "radon_cuda_backward", ([&] {
			backprojection_gpu<scalar_t>(
				tomo.data_ptr<scalar_t>(),
				sino.data_ptr<float>(),
				width,
				height,
				width_sino,
				n_angles,
				angles.data_ptr<float>(),
				x_center,
				y_center,
				u_center);
	}));

	return tomo;
}

