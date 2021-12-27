#include <opencv2/cvconfig.h>

#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/rgbd.hpp>

#include "tftn/readexr.h"
#include "tftn/tftn.h"

std::string input_file = std::string(INPUT_FILE) + "/0001.exr";
void ShowNormal(std::string win, cv::Mat result) {
  cv::Mat output(result.rows, result.cols, CV_16UC3);
  for (int i = 0; i < result.rows; ++i) {
    for (int j = 0; j < result.cols; ++j) {
      result.at<cv::Vec3f>(i, j) =
          result.at<cv::Vec3f>(i, j) / cv::norm(result.at<cv::Vec3f>(i, j));
      if (result.at<cv::Vec3f>(i, j)[2] < 0) {
        result.at<cv::Vec3f>(i, j) = -result.at<cv::Vec3f>(i, j);
      }
      output.at<cv::Vec3w>(i, j)[2] =
          (result.at<cv::Vec3f>(i, j)[0] + 1) * (65535 / 2.0);
      output.at<cv::Vec3w>(i, j)[1] =
          (result.at<cv::Vec3f>(i, j)[1] + 1) * (65535 / 2.0);
      output.at<cv::Vec3w>(i, j)[0] =
          (result.at<cv::Vec3f>(i, j)[2] + 1) * (65535 / 2.0);
    }
  }
  cv::imshow(win.c_str(), output);
  cv::waitKey(-1);
}

int main() {
  // camera's K matrix
  cv::Matx33d K(1056, 0, 1920 / 2, 0, 1056, 1080 / 2, 0, 0, 1);
  // input depth image, you need to reinstall opencv(do not use apt-get install)
  // with EXR try to use : sudo apt-get install libopenexr*, then install opencv
  cv::Mat depth_image, range_image;
  // depth_image = cv::imread(input_file, cv::IMREAD_UNCHANGED |
  // cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
  std::vector<cv::Mat> image;
  std::vector<std::string> channel_names;
  readexr(input_file, image, channel_names);  // load a exr image.
  std::cout << "image size" << image.size() << std::endl;
  cv::Mat gtx(image.at((3))), gty(-image.at(2)), gtz(image.at(1)),
      tmp;  // normal ground truth
  cv::merge(std::vector<cv::Mat>{gtx, gty, gtz}, tmp);

  // find the Z channel, copyt it to 'depth_image'
  for (int i = 0; i < channel_names.size(); ++i) {
    // std::cout<<channel_names.at(i);
    // std::cout<<image.size()<<std::endl;
    if (channel_names.at(i) == "Z") {
      depth_image = image.at(i);
    }
  }
  cv::rgbd::depthTo3d(depth_image, K, range_image);
  cv::Mat result;
  TFTN(range_image, K, R_MEDIAN_STABLE_8, &result);

  ShowNormal("ground truth result", tmp);
  ShowNormal("tftn result", result);
  return 0;
}