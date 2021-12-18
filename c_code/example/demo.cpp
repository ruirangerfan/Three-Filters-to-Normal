#include <cstdlib>
#include <iostream>
#include <eigen3/Eigen/Core>
#include "tftn/tftn.h"
#include "cvrgbd/rgbd.hpp"


/**
  * @brief Read depth images (.bin files)
  * */
cv::Mat LoadDepthImage(const std::string &path, const size_t width = 640,
                       const size_t height = 480){
  const int buffer_size = sizeof(float) * height * width;
  //char *buffer = new char[buffer_size];

  cv::Mat mat(cv::Size(width, height), CV_32FC1);

  // open filestream && read buffer
  std::ifstream fs_bin_(path, std::ios::binary);
  fs_bin_.read(reinterpret_cast<char*>(mat.data), buffer_size);
  fs_bin_.close();
  return mat;
}

int main(){
  int n; //the number of depth images.
  std::string param = "../data/android/params.txt";
  FILE *f = fopen(param.c_str(), "r");

  cv::Matx33d camera(0,0,0,0,0,0,0,0,1);
  fscanf(f, "%lf %lf %lf %lf %d", &camera(0,0),
         &camera(1,1), &camera(0, 2), &camera(1,2), &n);
  camera(0,2)--;  camera(1,2)--;

  auto depth_image = LoadDepthImage("../data/android/depth/000001.bin", 640, 480);
  cv::Mat_<float> s(depth_image);
  for (auto &it : s){
    if (fabs(it) < 1e-7){ //If the value equals 0, the point is infinite
      it = 1e10;
    }
  }

  //convert depth image to range image. watch out the problem of bgr and rgb.
  cv::rgbd::depthTo3d(depth_image, camera, range_image);
  std::vector<cv::Mat> matpart(3);
  cv::split(range_image, matpart);
  result.create(matpart[0].rows, matpart[0].cols, CV_32FC3);

  /*******************the core code*********************/
  TFTN(range_image, camera, R_MEANS_4, &result);
  /*****************************************/

  output.create(result.rows, result.cols, CV_16UC3);
  for (int i = 0; i < result.rows; ++ i){
    for (int j = 0; j < result.cols; ++ j){
      result.at<cv::Vec3f>(i, j) = result.at<cv::Vec3f>(i, j) / cv::norm(result.at<cv::Vec3f>(i, j));
      if (result.at<cv::Vec3f>(i, j)[2] < 0) {
        result.at<cv::Vec3f>(i, j) = -result.at<cv::Vec3f>(i, j);
      }
      output.at<cv::Vec3w>(i, j)[2] = (result.at<cv::Vec3f>(i, j)[0]+1)*(65535/2.0);
      output.at<cv::Vec3w>(i, j)[1] = (result.at<cv::Vec3f>(i, j)[1]+1)*(65535/2.0);
      output.at<cv::Vec3w>(i, j)[0] = (result.at<cv::Vec3f>(i, j)[2]+1)*(65535/2.0);
    }
  }
  cv::imshow("result", output);
  cv::waitKey(-1);
  return 0;
}

