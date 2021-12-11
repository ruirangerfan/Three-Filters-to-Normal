// Created by Bohuan Xue on 2019/10/18.

#ifndef TFTN_H_
#define TFTN_H_

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>
#include <VCL/vectorclass.h>
#include <iostream>


enum TFTN_METHOD{R_MEANS_8,
  R_MEDIAN_FAST_8,
  R_MEDIAN_STABLE_8,
  R_MEANS_4,
  R_MEDIAN_4,
  R_MEDIAN_FAST_4_8,
  R_MEDIAN_STABLE_4_8,
  R_MEANS_4_8,
  R_MEANS_SOBEL,
  R_MEDIAN_SOBEL,
  R_MEANS_SCHARR,
  R_MEDIAN_SCHARR,
  R_MEANS_PREWITT,
  R_MEDIAN_PREWITT};


void TFTN(const cv::Mat &range_image,
                          const cv::Matx33d camera,
                        const TFTN_METHOD method,
                          cv::Mat* output);


/* {
  const Vec8f kernel_x(-1, 0, 1, -2, 2, -1, 0, 1);
  const Vec8f kernel_y(-1, -2, -1, 0, 0, 1, 2, 1);
  const Vec4f kernel_x4(0, -1, 1, 0);
  const Vec4f kernel_y4(-1, 0, 0, 1);
  const Vec8f kernel_x48(0, 0, 0, -1, 1, 0, 0, 0);
  const Vec8f kernel_y48(0, -1, 0, 0, 0, 0, 1, 0);


  const Vec8f kernel_sobel_x(-1, 0, 1, -2, 2, -1, 0, 1);
  const Vec8f kernel_sobel_y(-1, -2, -1, 0, 0, 1, 2, 1);
  const Vec8f kernel_scharr_x(-3, 0, 3, -10, 10, -3, 0, 3);
  const Vec8f kernel_scharr_y(-3, -10, -3, 0, 0, 3, 10, 3);
  const Vec8f kernel_prewitt_x(-1, 0, 1, -1, 1, -1, 0, 1);
  const Vec8f kernel_prewitt_y(-1, -1, -1, 0, 0, 1, 1, 1);

  switch (method){
    case R_MEANS_8 :
      TFTN_MEAN(-kernel_x * camera(0,0), -kernel_y*camera(1,1), range_image, *output);
      break;
    case R_MEDIAN_FAST_8 :
      TFTN_MEDIAN_FAST(-kernel_x, -kernel_y, camera, range_image, *output);
      break;
    case R_MEDIAN_STABLE_8 :
      TFTN_MEDIAN_STABLE(-kernel_x, -kernel_y, camera, range_image, *output);
      break;
    case R_MEANS_4:
      TFTN_MEAN(-kernel_x4, -kernel_y4, camera, range_image, *output);
      break;
    case R_MEDIAN_4:
      TFTN_MEDIAN(-kernel_x4, -kernel_y4, camera, range_image, *output);
      break;
    case R_MEDIAN_FAST_4_8:
      TFTN_MEDIAN_FAST(-kernel_x48, -kernel_y48, camera, range_image, *output);
      break;
    case R_MEDIAN_STABLE_4_8:
      TFTN_MEDIAN_STABLE(-kernel_x48, -kernel_y48, camera, range_image, *output);
      break;
    case R_MEANS_4_8:
      TFTN_MEAN(-kernel_x48*camera(0,0), -kernel_y48*camera(1,1), range_image, *output);
      break;
    case R_MEANS_SOBEL:
      TFTN_MEAN(-kernel_sobel_x*camera(0,0), -kernel_sobel_y*camera(1,1), range_image, *output);
      break;
    case R_MEDIAN_SOBEL:
      TFTN_MEDIAN_STABLE(-kernel_sobel_x, -kernel_sobel_y, camera, range_image, *output);
      break;
    case R_MEANS_SCHARR:
      TFTN_MEAN(-kernel_scharr_x*camera(0,0), -kernel_scharr_y*camera(1,1), range_image, *output);
      break;
    case R_MEDIAN_SCHARR:
      TFTN_MEDIAN_STABLE(-kernel_scharr_x, -kernel_scharr_y, camera, range_image, *output);
      break;
    case R_MEANS_PREWITT:
      TFTN_MEAN(-kernel_prewitt_x*camera(0,0), -kernel_prewitt_y*camera(1,1), range_image, *output);
      break;
    case R_MEDIAN_PREWITT:
      TFTN_MEDIAN_STABLE(-kernel_prewitt_x, -kernel_prewitt_y, camera, range_image, *output);
      break;
    default:
      std::cerr<<"something wrong?" << std::endl;
      exit(-1);
  }
}*/






#endif //TFTN_RIN_H_

