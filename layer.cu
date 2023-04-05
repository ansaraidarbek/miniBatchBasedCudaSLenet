#include "layer.h"

__managed__ float learningRate = 1.0E-01f;
#define miniBatch 32

// Constructor
Layer::Layer(int M, int N, int O)
{
	this->M = M;
	this->N = N;
	this->O = O;

	int i;

	output = NULL;
	preact = NULL;
	bias = NULL;
	weight = NULL;
	h_bias = (float*)malloc(N * sizeof(float));
	h_weight = (float*)malloc(M * N * sizeof(float));

	for (i = 0; i < N; ++i) {
		h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
		/*h_bias[i] = 0.0f;*/
	}

	for (int j = 0; j < M * N; ++j) {
		h_weight[j] = 0.5f - float(rand()) / float(RAND_MAX);
		/*h_weight[i][j] = 0.05f;*/
	}

	cudaMalloc(&output, sizeof(float) * miniBatch * O);
	cudaMalloc(&preact, sizeof(float) * miniBatch * O);

	cudaMalloc(&bias, sizeof(float) * N);

	cudaMalloc(&weight, sizeof(float) * M * N);

	cudaMalloc(&d_output, sizeof(float) * O);
	cudaMalloc(&d_preact, sizeof(float) * O);
	cudaMalloc(&d_weight, sizeof(float) * M * N);

	cudaMemcpy(bias, h_bias, sizeof(float) * N, cudaMemcpyHostToDevice);

	cudaMemcpy(weight, h_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);

	free(h_bias);
	free(h_weight);
}

// Destructor
Layer::~Layer()
{
	cudaFree(output);
	cudaFree(preact);

	cudaFree(bias);

	cudaFree(weight);

	cudaFree(d_output);
	cudaFree(d_preact);
	cudaFree(d_weight);
}

// Send data one row from dataset to the GPU
void Layer::setOutput(float* data, int mB)
{
	cudaMemcpy(output, data, sizeof(float) *mB* O, cudaMemcpyHostToDevice);
}

// Reset GPU memory between iterations
void Layer::clear()
{
	cudaMemset(output, 0x00, sizeof(float) * miniBatch * O);
	cudaMemset(preact, 0x00, sizeof(float) * miniBatch * O);
}

void Layer::bp_clear()
{
	cudaMemset(d_output, 0x00, sizeof(float) * O);
	cudaMemset(d_preact, 0x00, sizeof(float) * O);
	cudaMemset(d_weight, 0x00, sizeof(float) * M * N);
}

__global__ void changelRate(float lr) {
	learningRate = lr;
}

__device__ float step_function(float v)
{
	return 1 / (1 + exp(-v));
}

__global__ void apply_step_function(float* input, float* output, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;
	const int y = blockIdx.y;

	for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx) {
		output[y*N+idx] = step_function(input[y*N+idx]);
	}
}

__global__ void makeError(float* err, float output[][10], unsigned int Y, const int N)
{
	const int pos = blockIdx.x;
	const int y = threadIdx.x;
	const int sizes = blockDim.x;
	float temp = 0;
	temp = (2 * ((Y == pos ? 1.0f : 0.0f) - output[y][pos]));
	atomicAdd(&err[pos], temp / sizes);
}

__global__ void apply_grad(float* output, float* grad, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;
	const int y = blockIdx.y;
	for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx) {
		output[y * N + idx] += learningRate * grad[y * N + idx];
	}
}

__global__ void fp_preact_c1(float input[][28][28], float preact[][6][24][24], float weight[6][5][5])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;
	const int y = blockIdx.y;

	const int N = 5 * 5 * 6 * 24 * 24;
	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1) % 5);
		const int i2 = ((idx /= 5) % 5);
		const int i3 = ((idx /= 5) % 6);
		const int i4 = ((idx /= 6) % 24);
		const int i5 = ((idx /= 24) % 24);

		atomicAdd(&preact[y][i3][i4][i5], weight[i3][i1][i2] * input[y][i4 + i1][i5 + i2]);
	}
	//__syncthreads();
	//if (y == 3 && pos == 0) {
	//	for (int i = 0; i < 6; i++) {
	//		for (int j = 0; j < 24; j++) {
	//			for (int k = 0; k < 24; k++) {
	//				printf("%.4f|", preact[y][i][j][k]);
	//			}
	//			printf("\n");
	//		}
	//		printf("\n");
	//		printf("\n");
	//	}
	//}
}

__global__ void fp_bias_c1(float preact[][6][24][24], float bias[6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;
	const int y = blockIdx.y;

	const int N = 6 * 24 * 24;

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1) % 6);
		const int i2 = ((idx /= 6) % 24);
		const int i3 = ((idx /= 24) % 24);

		preact[y][i1][i2][i3] += bias[i1];
	}
}

__global__ void fp_preact_s1(float input[][6][24][24], float preact[][6][6][6], float weight[1][4][4])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;
	const int y = blockIdx.y;

	const int N = 4 * 4 * 6 * 6 * 6;

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1) % 4);
		const int i2 = ((idx /= 4) % 4);
		const int i3 = ((idx /= 4) % 6);
		const int i4 = ((idx /= 6) % 6);
		const int i5 = ((idx /= 6) % 6);

		atomicAdd(&preact[y][i3][i4][i5], weight[0][i1][i2] * input[y][i3][i4 * 4 + i1][i5 * 4 + i2]);
	}
}

__global__ void fp_bias_s1(float preact[][6][6][6], float bias[1])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;
	const int y = blockIdx.y;

	const int N = 6 * 6 * 6;

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1) % 6);
		const int i2 = ((idx /= 6) % 6);
		const int i3 = ((idx /= 6) % 6);

		preact[y][i1][i2][i3] += bias[0];
	}
}

__global__ void fp_preact_f(float input[][6][6][6], float preact[][10], float weight[10][6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;
	const int y = blockIdx.y;

	const int N = 10 * 6 * 6 * 6;

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1) % 10);
		const int i2 = ((idx /= 10) % 6);
		const int i3 = ((idx /= 6) % 6);
		const int i4 = ((idx /= 6) % 6);

		atomicAdd(&preact[y][i1], weight[i1][i2][i3][i4] * input[y][i2][i3][i4]);
	}
}

__global__ void fp_bias_f(float preact[][10], float bias[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;
	const int y = blockIdx.y;

	const int N = 10;

	for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx) {
		preact[y][idx] += bias[idx];
	}
}

__global__ void bp_preact_f(float d_preact[10], float d_output[10], float preact[][10]) {
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10;

	for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx) {
		const float o = step_function(preact[0][idx]);
		d_preact[idx] = d_output[idx] * o * (1 - o);
	}
}

__global__ void bp_weight_f(float d_weight[10][6][6][6], float d_preact[10], float p_output[][6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10 * 6 * 6 * 6;

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1) % 10);
		const int i2 = ((idx /= 10) % 6);
		const int i3 = ((idx /= 6) % 6);
		const int i4 = ((idx /= 6) % 6);

		d_weight[i1][i2][i3][i4] = d_preact[i1] * p_output[0][i2][i3][i4];
	}
}

__global__ void bp_bias_f(float bias[10], float d_preact[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10;

	for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx) {
		bias[idx] += learningRate * d_preact[idx];
	}
}

__global__ void bp_output_s1(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10 * 6 * 6 * 6;

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1) % 10);
		const int i2 = ((idx /= 10) % 6);
		const int i3 = ((idx /= 6) % 6);
		const int i4 = ((idx /= 6) % 6);

		atomicAdd(&d_output[i2][i3][i4], n_weight[i1][i2][i3][i4] * nd_preact[i1]);
	}
}

__global__ void bp_preact_s1(float d_preact[6][6][6], float d_output[6][6][6], float preact[][6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6 * 6 * 6;

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1) % 6);
		const int i2 = ((idx /= 6) % 6);
		const int i3 = ((idx /= 6) % 6);
		const float o = step_function(preact[0][i1][i2][i3]);
		d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
	}
}

__global__ void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[][6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1 * 4 * 4 * 6 * 6 * 6;
	const float d = pow(6.0f, 3.0f);

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1) % 1);
		const int i2 = ((idx /= 1) % 4);
		const int i3 = ((idx /= 4) % 4);
		const int i4 = ((idx /= 4) % 6);
		const int i5 = ((idx /= 6) % 6);
		const int i6 = ((idx /= 6) % 6);

		atomicAdd(&d_weight[i1][i2][i3], (d_preact[i4][i5][i6] * p_output[0][i4][i5 * 4 + i2][i6 * 4 + i3]) / d);
	}
}

__global__ void bp_bias_s1(float bias[1], float d_preact[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6 * 6 * 6;
	const float d = pow(6.0f, 3.0f);

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1) % 6);
		const int i2 = ((idx /= 6) % 6);
		const int i3 = ((idx /= 6) % 6);

		atomicAdd(&bias[0], learningRate * d_preact[i1][i2][i3] / d);
	}
}

__global__ void bp_output_c1(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1 * 4 * 4 * 6 * 6 * 6;

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1) % 1);
		const int i2 = ((idx /= 1) % 4);
		const int i3 = ((idx /= 4) % 4);
		const int i4 = ((idx /= 4) % 6);
		const int i5 = ((idx /= 6) % 6);
		const int i6 = ((idx /= 6) % 6);

		atomicAdd(&d_output[i4][i5 * 4 + i2][i6 * 4 + i3], n_weight[i1][i2][i3] * nd_preact[i4][i5][i6]);
	}
}

__global__ void bp_preact_c1(float d_preact[6][24][24], float d_output[6][24][24], float preact[][6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6 * 24 * 24;

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1) % 6);
		const int i2 = ((idx /= 6) % 24);
		const int i3 = ((idx /= 24) % 24);
		const float o = step_function(preact[0][i1][i2][i3]);
		d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
	}
}

__global__ void bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[][28][28])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6 * 5 * 5 * 24 * 24;
	const float d = pow(24.0f, 2.0f);

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1) % 6);
		const int i2 = ((idx /= 6) % 5);
		const int i3 = ((idx /= 5) % 5);
		const int i4 = ((idx /= 5) % 24);
		const int i5 = ((idx /= 24) % 24);

		atomicAdd(&d_weight[i1][i2][i3], d_preact[i1][i4][i5] * p_output[0][i4 + i2][i5 + i3] / d);
	}
}

__global__ void bp_bias_c1(float bias[6], float d_preact[6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6 * 24 * 24;
	const float d = pow(24.0f, 2.0f);

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1) % 6);
		const int i2 = ((idx /= 6) % 24);
		const int i3 = ((idx /= 24) % 24);

		atomicAdd(&bias[i1], learningRate * d_preact[i1][i2][i3] / d);
	}
}