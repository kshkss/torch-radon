#include <cassert>
#include <cstdint>
#include <cstdio>

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static constexpr int warp_size = 32;
static constexpr int n_threads = 256;
static constexpr int n_warps = n_threads / warp_size;

__device__ __inline__
static float y_rotated(float theta, float x, float y)
{
	return sin(theta) * x + cos(theta) * y;
}

__device__ __inline__
static void vrange(float *vmin, float *vmax, float theta, float x0, float x1, float y0, float y1)
{
	for(; theta < 0; theta += 2.0 * M_PI){}
	for(; theta >= 2.0 * M_PI; theta -= 2.0 * M_PI){}

	if(theta < 0.5 * M_PI){
		*vmin = y_rotated(theta, x0, y0);
		*vmax = y_rotated(theta, x1, y1);
	}else if(theta < M_PI){
		*vmin = y_rotated(theta, x0, y1);
		*vmax = y_rotated(theta, x1, y0);
	}else if(theta < 1.5 * M_PI){
		*vmin = y_rotated(theta, x1, y1);
		*vmax = y_rotated(theta, x0, y0);
	}else{
		*vmin = y_rotated(theta, x1, y0);
		*vmax = y_rotated(theta, x0, y1);
	}
}

template<class scalar_t>
__device__ __inline__
static void projection(scalar_t *acc,
		cudaTextureObject_t texObj, int width, int height,
		float xc, float yc, float theta,
		float u, float v){
	float x_ = xc + cos(theta) * u + sin(theta) * v;
	float y_ = yc - sin(theta) * u + cos(theta) * v;

	if(x_ >= 0 && x_ <= (float)width - 1){
		if(y_ >= 0 && y_ <= (float)height - 1){
			*acc += static_cast<scalar_t>(tex2D<float>(texObj, x_, y_));
		}
	}
}

template<class scalar_t>
__global__
static void radonT_gpu_calc(scalar_t *sino, cudaTextureObject_t tomo,
		int width, int height, int umax,
		int n_angles, float *angles,
		float xc, float yc, float uc)
		
{
	int u = blockDim.x * blockIdx.x + threadIdx.x;
	int t = blockIdx.y;
	float theta = angles[t];
	float v_min, v_max;
	vrange(&v_min, &v_max, theta, -xc, (float)(width-1)-xc, -yc, (float)(height-1)-yc);

	scalar_t acc = 0.0;
	for(float v = v_min; v < v_max; v += 1.0){
		projection<scalar_t>(&acc, tomo, width, height, xc, yc, theta, (float)u - uc, v);
	}

	if(u < umax){
		sino[umax * t + u] = acc;
	}
}

#define CHECK(call)                                   \
{                                                     \
	const cudaError_t error = call;                   \
	if (error != cudaSuccess)                         \
	{                                                 \
		printf("Error: %s:%d,  ", __FILE__, __LINE__); \
		printf("code:%d, reason: %s\n", error,         \
			cudaGetErrorString(error));               \
		exit(1);                                       \
	}                                                 \
}

template<class scalar_t>
void radonT_gpu(scalar_t *sino, const float *tomo,
		int width, int height, int umax,
		int n_angles, float *angles,
		float xc, float yc, float uc)
{
	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray *tomo_;
	cudaMallocArray(&tomo_, &channelDesc, width, height);
	CHECK(cudaMemcpy2DToArray(tomo_, 0, 0, tomo, width*sizeof(float), width*sizeof(float), height, cudaMemcpyDeviceToDevice));

	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = tomo_;

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
	dim3 grid((width + block.x - 1) / block.x, n_angles, 1);
	radonT_gpu_calc<scalar_t><<<grid, block>>>(sino, texObj, width, height, umax, n_angles, angles, xc, yc, uc);

	cudaDestroyTextureObject(texObj);
	cudaFreeArray(tomo_);
}

torch::Tensor radon_cuda_forward(
		torch::Tensor sino,
		torch::Tensor tomo,
		torch::Tensor angles,
		float x_center,
		float y_center,
		float u_center)
{
	int height = tomo.size(0);
	int width = tomo.size(1);
	int n_angles = sino.size(0);
	int width_sino = sino.size(1);

	AT_DISPATCH_FLOATING_TYPES(sino.type(), "radon_cuda_forward", ([&] {
			radonT_gpu<scalar_t>(
				sino.data_ptr<scalar_t>(),
				tomo.data_ptr<float>(),
				width,
				height,
				width_sino,
				n_angles,
				angles.data_ptr<float>(),
				x_center,
				y_center,
				u_center);
	}));

	return sino;
}
