#include <omp.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include "opencv2/opencv.hpp"

#define uo 350
#define vo 200
#define fx 1400
#define fy 1400
#define vmax 480
#define umax 640
#define offset 600

enum dataset { torusknot };

inline void check_gpu_compute_capability(void)
{
	/// <<<Obtain the information of the devices>>>
	std::cout << ">>> GPU Properties" << std::endl << std::endl;
	int deviceCount;
	// Declare a variable to enumerate the devices
	cudaGetDeviceCount(&deviceCount);
	// Get the number of the device(s)
	std::cout << "The host system has " << deviceCount << " device(s)" << std::endl;
	std::cout << std::endl;
	cudaDeviceProp deviceProp;
	// Declare a variable to get the properties of the cuda device
	for (int device = 0; device < deviceCount; device++)
	{
		cudaGetDeviceProperties(&deviceProp, device);
		// The first variable is the address of the deviceProp
		// The second variable is the device we want to enumerate
		std::cout << deviceProp.name << ": " << std::endl;
		std::cout << "The compute capability " << deviceProp.major << "." << deviceProp.minor << ", " << deviceProp.multiProcessorCount << " Multi-Processors." << std::endl;
		// deviceProp.multiProcessorCount is used to get the number of multiprocessors 
	}
	std::cout << std::endl << "GPU Properties <<<" << std::endl << std::endl << std::endl;
	/// <<<Obtain the information of the devices>>>
}

inline int idivup(int a, int b){
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

inline void load_data(
	dataset dataname,
	const int frm,
	float* X,
	float* Y,
	float* Z,
	float* D,
	char* M)
{
	char data_dir[256];
	char frm_num2str[20];
	char dataset_name[50];
	strcpy_s(data_dir, "..//..//data//");
	sprintf(frm_num2str, "%06d", frm);
	if (dataname == torusknot){
		strcat_s(data_dir, "torusknot//depth//");
		strcat_s(data_dir, frm_num2str);
		strcat_s(data_dir, ".bin");
	}
	// open filestream && read buffer
	cv::Mat data_mat(cv::Size(umax, vmax), CV_32F);

	std::ifstream bin_file(data_dir, std::ios::binary);
	bin_file.read(reinterpret_cast<char*>(data_mat.data), sizeof(float) * vmax * umax);
	bin_file.close();

	for (int i = 0; i < vmax; i++) {
		for (int j = 0; j < umax; j++) {
			int idx = i * umax + j;
			Z[idx] = offset * data_mat.at<float>(i, j);
			D[idx] = 1 / Z[idx];
			X[idx] = Z[idx] * (j + 1 - uo) / fx;
			Y[idx] = Z[idx] * (i + 1 - vo) / fy;
			if (Z[idx] == offset) {
				M[idx] = 255;
			}
			else {
				M[idx] = 0;
			}
		}
	}
}


inline void output_visualization(
	float* nx, float* ny, float* nz, cv::Mat& vis
) {

	for (int i = 0; i < vmax; i++)
	{
		for (int j = 0; j < umax; j++) {
			const int idx = i * umax + j;
			if (!isnan(nx[idx]) && !isnan(ny[idx]) && !isnan(nz[idx]))
			{
				const short nx_vis = (short)(nx[idx] * 65535);
				const short ny_vis = (short)(ny[idx] * 65535);
				const short nz_vis = (short)(nz[idx] * 65535);


				vis.at<cv::Vec3s>(i, j)[0] = nx_vis;
				vis.at<cv::Vec3s>(i, j)[1] = ny_vis;
				vis.at<cv::Vec3s>(i, j)[2] = nz_vis;
			}
			else {

				vis.at<cv::Vec3s>(i, j)[0] = 1;
				vis.at<cv::Vec3s>(i, j)[1] = 1;
				vis.at<cv::Vec3s>(i, j)[2] = 1;


			}


		}
	}
}









