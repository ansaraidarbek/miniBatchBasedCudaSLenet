#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "cublas_api.h"
#include <stdio.h>

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "layer.h"

#include <cuda.h>
#include <cstdio>
#include <time.h>

#include <cstdlib>
#include <vector>
#include <memory>

// Define layers of CNN
static Layer l_input = Layer(0, 0, 28 * 28);
static Layer l_c1 = Layer(5 * 5, 6, 24 * 24 * 6);
static Layer l_s1 = Layer(4 * 4, 1, 6 * 6 * 6);
static Layer l_f = Layer(6 * 6 * 6, 10, 10);

static mnist_data* train_set, * test_set;
static unsigned int train_cnt, test_cnt;

static void learn(int epochs, int mB);
static unsigned int classify(float data[28][28]);
static void test();
static double forward_pass(float Arr[][28][28], int mB);
static double back_pass(int pos);
double max_acc = 0;
dim3 gridSize(64, 1);
dim3 blockSize(64, 1);

static inline void loaddata()
{
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}

int main(int argc, const  char** argv)
{
	srand(time(NULL));

	if (argc > 1) {
		int epochs = atoi(argv[1]);
		int mB = atoi(argv[2]);
		loaddata();
		learn(epochs, mB);

	}
	else {
		printf("# epocs, activation function should be passed");
	}

	return 0;
}

// Forward propagation of a single row in dataset
static double forward_pass(float Arr[][28][28], int mB)
{
	l_input.clear();
	l_c1.clear();
	l_s1.clear();
	l_f.clear();

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	l_input.setOutput((float*)Arr, mB);
	cudaEventRecord(start);
	dim3 gridSize2(64, mB);
	fp_preact_c1 << <gridSize2, blockSize >> > ((float(*)[28][28])l_input.output, (float(*)[6][24][24])l_c1.preact, (float(*)[5][5])l_c1.weight);
	fp_bias_c1 << <gridSize2, blockSize >> > ((float(*)[6][24][24])l_c1.preact, l_c1.bias);
	apply_step_function << <gridSize2, blockSize >> > (l_c1.preact, l_c1.output, l_c1.O);

	fp_preact_s1 << <gridSize2, blockSize >> > ((float(*)[6][24][24])l_c1.output, (float(*)[6][6][6])l_s1.preact, (float(*)[4][4])l_s1.weight);
	fp_bias_s1 << <gridSize2, blockSize >> > ((float(*)[6][6][6])l_s1.preact, l_s1.bias);
	apply_step_function << <gridSize2, blockSize >> > (l_s1.preact, l_s1.output, l_s1.O);

	fp_preact_f << <gridSize2, blockSize >> > ((float(*)[6][6][6])l_s1.output, (float(*)[10])l_f.preact, (float(*)[6][6][6])l_f.weight);
	fp_bias_f << <gridSize2, blockSize >> > ((float(*)[10])l_f.preact, l_f.bias);
	apply_step_function << <gridSize2, blockSize >> > (l_f.preact, l_f.output, l_f.O);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	return milliseconds;
}

// Back propagation to update weights
static double back_pass(int pos)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int total = 5;
	cudaEventRecord(start);

	bp_preact_f << <gridSize, blockSize >> > (l_f.d_preact, l_f.d_output, (float(*)[10])l_f.preact);
	bp_weight_f << <gridSize, blockSize >> > ((float(*)[6][6][6])l_f.d_weight, l_f.d_preact, (float(*)[6][6][6])l_s1.output);
	bp_bias_f << <gridSize, blockSize >> > (l_f.bias, l_f.d_preact);
	bp_output_s1 << <gridSize, blockSize >> > ((float(*)[6][6])l_s1.d_output, (float(*)[6][6][6])l_f.weight, l_f.d_preact);
	bp_preact_s1 << <gridSize, blockSize >> > ((float(*)[6][6])l_s1.d_preact, (float(*)[6][6])l_s1.d_output, (float(*)[6][6][6])l_s1.preact);
	bp_weight_s1 << <gridSize, blockSize >> > ((float(*)[4][4])l_s1.d_weight, (float(*)[6][6])l_s1.d_preact, (float(*)[6][24][24])l_c1.output);
	bp_bias_s1 << <gridSize, blockSize >> > (l_s1.bias, (float(*)[6][6])l_s1.d_preact);
	bp_output_c1 << <gridSize, blockSize >> > ((float(*)[24][24])l_c1.d_output, (float(*)[4][4])l_s1.weight, (float(*)[6][6])l_s1.d_preact);
	bp_preact_c1 << <gridSize, blockSize >> > ((float(*)[24][24])l_c1.d_preact, (float(*)[24][24])l_c1.d_output, (float(*)[6][24][24])l_c1.preact);
	bp_weight_c1 << <gridSize, blockSize >> > ((float(*)[5][5])l_c1.d_weight, (float(*)[24][24])l_c1.d_preact, (float(*)[28][28])l_input.output);
	bp_bias_c1 << <gridSize, blockSize >> > (l_c1.bias, (float(*)[24][24])l_c1.d_preact);


	apply_grad << <gridSize, blockSize >> > (l_f.weight, l_f.d_weight, l_f.M * l_f.N);
	apply_grad << <gridSize, blockSize >> > (l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
	apply_grad << <gridSize, blockSize >> > (l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	return milliseconds;
}

static void learn(int epochs, int mB)
{
	static cublasHandle_t blas;
	cublasCreate(&blas);
	float err;
	float forward_time = 0.0;
	float backward_time = 0.0;
	float *Arr = (float*)malloc(mB*28*28 * sizeof(float));

	fprintf(stdout, "Learning\n");
	while (epochs < 0 || epochs-- > 0) {
		err = 0.0f;
		for (int i = 0; i < train_cnt; i=i+ mB) {
			float tmp_err;

			for (int j = i; j < i + mB; j++) {
				for (int k = 0; k < 28; k++) {
					for (int z = 0; z < 28; z++) {
						Arr[(j% mB)*28*28 + k *28 + z] = train_set[j].data[k][z];
					}
				}
			}

			forward_time += forward_pass((float(*)[28][28])Arr, mB);
			l_f.bp_clear();
			l_s1.bp_clear();
			l_c1.bp_clear();
			makeError << <10, mB >> > (l_f.d_output, (float(*)[10])l_f.output, train_set[i].label, 10);
			cublasSnrm2(blas, 10, l_f.d_output, 1, &tmp_err);
			err += tmp_err;

			backward_time += back_pass(i);
		}

		err /= train_cnt;

		fprintf(stdout, "%e\n", err);
		fprintf(stdout, "%f\n", forward_time);
		fprintf(stdout, "%f\n", backward_time);
		test();
	}

}

// Returns label of given data (0-9)
static unsigned int classify(float data[28][28])
{
	float res[10];

	forward_pass((float(*)[28][28])data, 1);

	unsigned int max = 0;

	cudaMemcpy(res, l_f.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 1; i < 10; ++i) {
		//printf("%f\n", res[i]);
		if (res[max] < res[i]) {
			max = i;
		}
	}
	//printf("\n");
	return max;
}

// Perform forward propagation of test data
static void test()
{
	int error = 0;

	for (int i = 0; i < test_cnt; ++i) {
		/*printf("%i, %i\n", classify(test_set[i].data), test_set[i].label);*/
		if (classify(test_set[i].data) != test_set[i].label) {
			++error;
		}
	}

	fprintf(stdout, "%e\n\n",
		double(error) / double(test_cnt) * 100.0);
}