#include <cuda.h>
#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "device_launch_parameters.h"


#define Block_x 32
#define Block_y 32

using namespace std;
using namespace cv;

texture<float, 2, cudaReadModeElementType> X_tex;
texture<float, 2, cudaReadModeElementType> Y_tex;
texture<float, 2, cudaReadModeElementType> Z_tex;
texture<float, 2, cudaReadModeElementType> D_tex;

enum kernel_type{SOBEL, PREWITT};
enum nz_filter_type {MEAN, MEDIAN};
enum normalization_type { POS, NEG };
enum visualization_type {OPEN, CLOSE};

__global__ void normal_estimation_sobel_median(
	float* nx_dev,
	float* ny_dev,
	float* nz_dev,
	float* Volume_dev,
	normalization_type normalization,
	visualization_type visualization) {

	int v = blockDim.y * blockIdx.y + threadIdx.y;
	int u = blockDim.x * blockIdx.x + threadIdx.x;

	if ((u >= 1) && (u < umax - 1) && (v >= 1) && (v < vmax - 1)) {

		const int idx0 = v * umax + u;

		const float nx = (2 * (tex2D(D_tex, u - 1, v) - tex2D(D_tex, u + 1, v))
			+ tex2D(D_tex, u - 1, v - 1) - tex2D(D_tex, u + 1, v - 1)
			+ tex2D(D_tex, u - 1, v + 1) - tex2D(D_tex, u + 1, v + 1)) * fx;

		const float ny = (2 * (tex2D(D_tex, u, v - 1) - tex2D(D_tex, u, v + 1))
			+ tex2D(D_tex, u - 1, v - 1) - tex2D(D_tex, u - 1, v + 1)
			+ tex2D(D_tex, u + 1, v - 1) - tex2D(D_tex, u + 1, v + 1)) * fy;

		nx_dev[idx0] = nx;
		ny_dev[idx0] = ny;

		const float X0 = tex2D(X_tex, u, v);
		const float Y0 = tex2D(Y_tex, u, v);
		const float Z0 = tex2D(Z_tex, u, v);

		float nz = 0;

		int valid_num = 0;
		float nz_sum = 0;

		for (int i = -1; i <= 1; i++) {
			for (int j = -1; j <= 1; j++) {
				const int tmp = i * umax + j;
				const int idx1 = idx0 + tmp;

				const float X1 = tex2D(X_tex, u + j, v + i);
				const float Y1 = tex2D(Y_tex, u + j, v + i);
				const float Z1 = tex2D(Z_tex, u + j, v + i);

				const float X_d = X0 - X1;
				const float Y_d = Y0 - Y1;
				const float Z_d = Z0 - Z1;

				if (Z0 != Z1) {
					const float nz = -(nx * X_d + ny * Y_d) / Z_d;
					if (nz <= 0) {
						valid_num++;
						Volume_dev[vmax * umax * valid_num + idx0] = nz;
					}
				}
			}
		}
		Volume_dev[idx0] = valid_num;

		if (valid_num == 1) {
			nz = Volume_dev[vmax * umax + idx0];
		}
		else if (valid_num == 2) {
			nz = (Volume_dev[vmax * umax + idx0] + Volume_dev[vmax * umax * 2 + idx0]) / 2;
		}
		else {
			for (int m = 1; m < valid_num; m++)
			{
				for (int n = 0; n < valid_num - m; n++)
				{
					const float nz_0 = Volume_dev[vmax * umax * (n + 1) + idx0];
					const float nz_1 = Volume_dev[vmax * umax * (n + 2) + idx0];
					if (nz_0 > nz_1)
					{
						Volume_dev[vmax * umax * (n + 1) + idx0] = nz_1;
						Volume_dev[vmax * umax * (n + 2) + idx0] = nz_0;
					}
				}
			}
			if (valid_num % 2 == 0)
			{
				nz = (Volume_dev[vmax * umax * (valid_num / 2) + idx0]
					+ Volume_dev[vmax * umax * (valid_num / 2 + 1) + idx0]) / 2;
			}
			else
			{
				nz = Volume_dev[vmax * umax * ((valid_num + 1) / 2) + idx0];
			}
		}
		if (normalization == POS) {
			float mag = sqrt(nx * nx + ny * ny + nz * nz);
			if (mag != 0) {
				nx_dev[idx0] = nx / mag;
				ny_dev[idx0] = ny / mag;
				nz_dev[idx0] = nz / mag;
			}
		}
		if (visualization == OPEN) {
			nx_dev[idx0] = (1 + nx_dev[idx0]) / 2;
			ny_dev[idx0] = (1 + ny_dev[idx0]) / 2;
			nz_dev[idx0] = (1 + nz_dev[idx0]) / 2;
		}
	}
}

__global__ void normal_estimation_bg_median(
	float* nx_dev,
	float* ny_dev,
	float* nz_dev,
	float* Volume_dev,
	normalization_type normalization,
	visualization_type visualization) {

	int v = blockDim.y * blockIdx.y + threadIdx.y;
	int u = blockDim.x * blockIdx.x + threadIdx.x;

	if ((u >= 1) && (u < umax - 1) && (v >= 1) && (v < vmax - 1)) {

		const int idx0 = v * umax + u;

		const float nx = (tex2D(D_tex, u - 1, v) - tex2D(D_tex, u + 1, v)) * fx;
		const float ny = (tex2D(D_tex, u, v - 1) - tex2D(D_tex, u, v + 1)) * fy;

		nx_dev[idx0] = nx;
		ny_dev[idx0] = ny;

		const float X0 = tex2D(X_tex, u, v);
		const float Y0 = tex2D(Y_tex, u, v);
		const float Z0 = tex2D(Z_tex, u, v);

		float nz = 0;

		int valid_num = 0;
		float nz_sum = 0;

		for (int i = -1; i <= 1; i++) {
			for (int j = -1; j <= 1; j++) {
				const int tmp = i * umax + j;
				const int idx1 = idx0 + tmp;

				const float X1 = tex2D(X_tex, u + j, v + i);
				const float Y1 = tex2D(Y_tex, u + j, v + i);
				const float Z1 = tex2D(Z_tex, u + j, v + i);

				const float X_d = X0 - X1;
				const float Y_d = Y0 - Y1;
				const float Z_d = Z0 - Z1;

				if (Z0 != Z1) {
					const float nz = -(nx * X_d + ny * Y_d) / Z_d;
					if (nz <= 0) {
						valid_num++;
						Volume_dev[vmax * umax * valid_num + idx0] = nz;
					}
				}
			}
		}
		Volume_dev[idx0] = valid_num;

		if (valid_num == 1) {
			nz = Volume_dev[vmax * umax + idx0];
		}
		else if (valid_num == 2) {
			nz = (Volume_dev[vmax * umax + idx0] + Volume_dev[vmax * umax * 2 + idx0]) / 2;
		}
		else {
			for (int m = 1; m < valid_num; m++)
			{
				for (int n = 0; n < valid_num - m; n++)
				{
					const float nz_0 = Volume_dev[vmax * umax * (n + 1) + idx0];
					const float nz_1 = Volume_dev[vmax * umax * (n + 2) + idx0];
					if (nz_0 > nz_1)
					{
						Volume_dev[vmax * umax * (n + 1) + idx0] = nz_1;
						Volume_dev[vmax * umax * (n + 2) + idx0] = nz_0;
					}
				}
			}
			if (valid_num % 2 == 0)
			{
				nz = (Volume_dev[vmax * umax * (valid_num / 2) + idx0]
					+ Volume_dev[vmax * umax * (valid_num / 2 + 1) + idx0]) / 2;
			}
			else
			{
				nz = Volume_dev[vmax * umax * ((valid_num + 1) / 2) + idx0];
			}
		}
		if (normalization == POS) {
			float mag = sqrt(nx * nx + ny * ny + nz * nz);
			if (mag != 0) {
				nx_dev[idx0] = nx / mag;
				ny_dev[idx0] = ny / mag;
				nz_dev[idx0] = nz / mag;
			}
		}
		if (visualization == OPEN) {
			nx_dev[idx0] = (1 + nx_dev[idx0]) / 2;
			ny_dev[idx0] = (1 + ny_dev[idx0]) / 2;
			nz_dev[idx0] = (1 + nz_dev[idx0]) / 2;
		}
	}
}


int main(int, char) {

	check_gpu_compute_capability();

	// Setting kernel and nz_filter types
	kernel_type kernel = SOBEL;
	nz_filter_type nz_filter = MEDIAN;
	normalization_type normalization = POS;
	visualization_type visualization = OPEN;


	float min_runtime = 100;
	float max_runtime = 0;



	// Setting parameters
	const int pixel_number = vmax * umax;

	// Create blocks and threads
	dim3 threads = dim3(Block_x, Block_y);
	dim3 blocks = dim3(idivup(umax, threads.x), idivup(vmax, threads.y));

	// compute memsize
	const int char_memsize = sizeof(char) * pixel_number;
	const int float_memsize = sizeof(float) * pixel_number;

	// declare eight arrays
	char* M = (char*)calloc(pixel_number, sizeof(char));
	float* D = (float*)calloc(pixel_number, sizeof(float));
	float* Z = (float*)calloc(pixel_number, sizeof(float));
	float* X = (float*)calloc(pixel_number, sizeof(float));
	float* Y = (float*)calloc(pixel_number, sizeof(float));
	float* nx = (float*)calloc(pixel_number, sizeof(float));
	float* ny = (float*)calloc(pixel_number, sizeof(float));
	float* nz = (float*)calloc(pixel_number, sizeof(float));

	cv::Mat M_mat(vmax, umax, CV_8U, M);
	cv::Mat D_mat(vmax, umax, CV_32F, D);
	cv::Mat X_mat(vmax, umax, CV_32F, X);
	cv::Mat Y_mat(vmax, umax, CV_32F, Y);
	cv::Mat Z_mat(vmax, umax, CV_32F, Z);
	cv::Mat nx_mat(vmax, umax, CV_32F, nx);
	cv::Mat ny_mat(vmax, umax, CV_32F, ny);
	cv::Mat nz_mat(vmax, umax, CV_32F, nz);


	// Bind X, Y, Z and D with texture memory;
	cudaChannelFormatDesc desc_X = cudaCreateChannelDesc<float>();
	cudaChannelFormatDesc desc_Y = cudaCreateChannelDesc<float>();
	cudaChannelFormatDesc desc_Z = cudaCreateChannelDesc<float>();
	cudaChannelFormatDesc desc_D = cudaCreateChannelDesc<float>();

	cudaArray* X_texture, * Y_texture, * Z_texture, * D_texture;

	cudaMallocArray(&X_texture, &desc_X, umax, vmax);
	cudaMallocArray(&Y_texture, &desc_Y, umax, vmax);
	cudaMallocArray(&Z_texture, &desc_Z, umax, vmax);
	cudaMallocArray(&D_texture, &desc_D, umax, vmax);

	// Create four arrays to store nx, ny, nz and volume;
	float* nx_dev, * ny_dev, * nz_dev, * Volume_dev;

	cudaMalloc((void**)& nx_dev, float_memsize);
	cudaMalloc((void**)& ny_dev, float_memsize);
	cudaMalloc((void**)& nz_dev, float_memsize);
	cudaMalloc((void**)& Volume_dev, float_memsize * 9);

	for (int frm = 1; frm <= 2500; frm++) {
		load_data(
			torusknot,
			frm,
			X,
			Y,
			Z,
			D,
			M);

	
		cudaMemcpyToArray(X_texture, 0, 0, X, float_memsize, cudaMemcpyHostToDevice);
		cudaMemcpyToArray(Y_texture, 0, 0, Y, float_memsize, cudaMemcpyHostToDevice);
		cudaMemcpyToArray(Z_texture, 0, 0, Z, float_memsize, cudaMemcpyHostToDevice);
		cudaMemcpyToArray(D_texture, 0, 0, D, float_memsize, cudaMemcpyHostToDevice);

		cudaBindTextureToArray(X_tex, X_texture, desc_X);
		cudaBindTextureToArray(Y_tex, Y_texture, desc_Y);
		cudaBindTextureToArray(Z_tex, Z_texture, desc_Z);
		cudaBindTextureToArray(D_tex, D_texture, desc_D);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, NULL);

		normal_estimation_bg_median << < blocks, threads >> > (
			nx_dev,
			ny_dev,
			nz_dev,
			Volume_dev,
			normalization,
			visualization);

		cudaEventRecord(stop, NULL);
		cudaEventSynchronize(stop);
		float msecTotal = 1.0f;
		cudaEventElapsedTime(&msecTotal, start, stop);
		std::cout << "runtime: " << msecTotal << std::endl;
		
		
		if (msecTotal < min_runtime){
			min_runtime = msecTotal;
		}
		if (msecTotal > max_runtime) {
			max_runtime = msecTotal;
		}



		cudaMemcpy(nx, nx_dev, float_memsize, cudaMemcpyDeviceToHost);
		cudaMemcpy(ny, ny_dev, float_memsize, cudaMemcpyDeviceToHost);
		cudaMemcpy(nz, nz_dev, float_memsize, cudaMemcpyDeviceToHost);

		
		
		cv::Mat vis_mat(vmax, umax, CV_16UC3);
		output_visualization(
			nx, ny, nz, vis_mat);
		

		std::cout << "finish" << endl;

		namedWindow("result", WINDOW_AUTOSIZE);
		imshow("result", vis_mat);
		waitKey(30);


		std::cout << frm << endl;
	}
	std::cout << std::endl << std::endl << "runtime: " << min_runtime << std::endl;
	return 0;
}