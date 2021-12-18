#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/cvconfig.h>
#include "tftn/tftn.h"

#include <stdexcept>
#include <string>
#include <memory>

#include <unistd.h>


#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfChannelList.h>


//#define throw(...)
#include <pcgImageIO/OpenEXRIO.h>

#ifndef HAVE_OPENEXR
#error NO_OPENEXR
#endif

std::string input_file = std::string(INPUT_FILE) + "/0001.exr" ;

//using namespace OPENEXR_IMF_INTERNAL_NAMESPACE;

using namespace Imf;
using Imath::Box2i;



//~/github/hdritools-0.5.0/OpenEXR_Matlab$ vi exrreadchannels.cpp


std::vector<std::vector<float>> frame_data;

// Prepares a framebuffer for the requested channels, allocating also the
// appropriate Matlab memory
void prepareFrameBuffer(FrameBuffer & fb, const Box2i & dataWindow,
                        const ChannelList & channels,
                        const std::vector<std::string> & requestedChannels){
    assert(!requestedChannels.empty());

    const Box2i & dw = dataWindow;
    const int width  = dw.max.x - dw.min.x + 1;
    const int height = dw.max.y - dw.min.y + 1;

    //分配内存
    frame_data.resize(requestedChannels.size());

    std::cout<<"channel数量"<<requestedChannels.size() << std::endl;
    for (int i = 0; i < requestedChannels.size(); ++ i){
        frame_data.at(i).resize((width*height) *sizeof(float));
        memset(frame_data.at(i).data(), 0, sizeof((width*height) *sizeof(float)));
    }


    // The "weird" strides are because Matlab uses column-major order
    //const int xStride = height;
    //const int yStride = 1;

    const int xStride = 1;
    const int yStride = width;

    // Offset for all the slices
    const off_t offset = - (dw.min.x * xStride + dw.min.y * yStride);


    for (size_t i = 0; i != requestedChannels.size(); ++i) {
        // Allocate the memory
        //mxArray * data =
        //        mxCreateNumericMatrix(height, width, mxSINGLE_CLASS, mxREAL);
        //outMatlabData[i] = data;
        //float * ptr = static_cast<float*>(mxGetData(data));
        // Get the appropriate sampling factors
        int xSampling = 1, ySampling = 1;
        ChannelList::ConstIterator cIt = channels.find(requestedChannels[i].c_str());
        if (cIt != channels.end()) {
            xSampling = cIt.channel().xSampling;
            ySampling = cIt.channel().ySampling;
        }else{
            std::cout<<"error"<<std::endl;
            exit(-1);

        }
        std::cout<<xStride <<" "<<xSampling << std::endl;

        // Insert the slice in the framebuffer
        fb.insert(requestedChannels[i].c_str(), Slice(FLOAT, (char*)(frame_data.at(i).data()),
                                                      sizeof(float) * xStride,
                                                      sizeof(float) * yStride,
                                                      xSampling, ySampling));
    }
}


// Utility to fill the given array with the names of all the channels
inline void getChannelNames(const ChannelList & channels,
                            std::vector<std::string> & result)
{
    typedef ChannelList::ConstIterator CIter;

    for (CIter it = channels.begin(); it != channels.end(); ++it) {
        result.push_back(std::string(it.name()));
    }
}


void io(){
    InputFile img(input_file.c_str());
    const ChannelList & channels = img.header().channels();

    std::vector<std::string> channelNames;

    // Prepare the framebuffer
    const Box2i & dw = img.header().dataWindow();
    const ChannelList & imgChannels = img.header().channels();

    getChannelNames(img.header().channels(), channelNames);
    for (int i = 0; i < channelNames.size(); ++ i){
        std::cout<< channelNames.at(i)<< std::endl;
    }


    FrameBuffer framebuffer;
    prepareFrameBuffer(framebuffer, dw, imgChannels, channelNames);
    // Actually read the pixels

    std::cout<<"@"<<std::endl;
    img.setFrameBuffer(framebuffer);  //设置好，

    std::cout<<"准备读取"<<std::endl;
    img.readPixels(dw.min.y, dw.max.y);

    std::cout<<"读取完毕"<<std::endl;

    const int width  = dw.max.x - dw.min.x + 1;
    const int height = dw.max.y - dw.min.y + 1;


    cv::Mat r(height ,width, CV_32F);
    memset(r.data, 0, sizeof((height*width)*sizeof(float)));

    std::cout<<"数据开始转移到cv mat格式上"<<std::endl;
    memmove(r.data, frame_data.at(3).data(), (height*width)*sizeof(float));
    std::cout<<"转移成功，显示图片"<<std::endl;
    cv::imshow("abc", r);
    cv::waitKey(-1);








    exit(-1);
}


/*
void io2(){

    pcg::RGBAImageSoA image;
    pcg::OpenEXRIO::Load(image, input_file.c_str());
    std::cout<<"the channel of image is :"<<image.NUM_CHANNELS << std::endl;
    std::cout << image.Width()  << std::endl;
    std::cout << image.Height() << std::endl;
    float* nx = image.GetDataPointer<pcg::RGBAImageSoA::R>();
    float* ny = image.GetDataPointer<pcg::RGBAImageSoA::G>();
    float* nz = image.GetDataPointer<pcg::RGBAImageSoA::B>();
    float* depth = image.GetDataPointer<pcg::RGBAImageSoA::A>();
    cv::Mat img(image.Height(), image.Width(), CV_32FC1);
    memmove(img.data, depth, image.Width() * image.Height() * sizeof(float));
    std::cout << img.at<float>(500,500)<<std::endl;
    for (int i = 0; i < image.Height() * image.Width(); ++ i){
        img.at<float>(i) = log(img.at<float>(i) + 1);
    }
    //cv::log(img, img);

    cv::imshow("asf", img);
    cv::waitKey(-1);


}
 */


int main(){
    io();
    //camera's K matrix
    cv::Matx33d K(1056, 0, 1920/2,
                  0, 1056, 1080/2,
                  0,    0,    1);

    //input depth image, you need to reinstall opencv(do not use apt-get install) with EXR
    //try to use : sudo apt-get install libopenexr*, then install opencv

    cv::Mat depth_image, range_image;
    //depth_image = cv::imread(input_file, cv::IMREAD_UNCHANGED | cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
    depth_image = cv::imread(input_file,  cv::IMREAD_ANYCOLOR );
    //cv::imreadmulti()
    std::cout<<depth_image.channels() << std::endl;

    //cv::rgbd::depthTo3d(depth_image, K, range_image);

    /*
  cv::Mat range_image;
  cv::Mat result;
  cv::Mat output;

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

  TFTN(range_image, camera, R_MEANS_4, &result);

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
  */
}

