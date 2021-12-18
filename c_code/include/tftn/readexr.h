//
// Created by zpmc on 2021/12/18.
//

#ifndef TFTN_READEXR_H
#define TFTN_READEXR_H



#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>

/**
 * @brief read .exr format
 * @param[in] path
 * @param[out] image
 * @parm[out] channel_names
 * */
void readexr(std::string path, std::vector<cv::Mat> image, std::vector<std::string> channel_name);




#endif //TFTN_READEXR_H
