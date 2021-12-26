//
// Created by zpmc on 2021/12/18.
//

#include "readexr.h"


#define throw(a)

#include "IexBaseExc.h"
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfChannelList.h>

#undef throw

using namespace Imf;
using Imath::Box2i;

static std::vector<std::vector<float>> frame_data;

// Prepares a framebuffer for the requested channels, allocating also the
// appropriate Matlab memory
void prepareFrameBuffer(FrameBuffer & fb,  const Box2i & dataWindow,
                        const ChannelList & channels,
                        std::vector<std::string> & requestedChannels){
    assert(!requestedChannels.empty());

    const Box2i & dw = dataWindow;
    const int width  = dw.max.x - dw.min.x + 1;
    const int height = dw.max.y - dw.min.y + 1;

    //分配内存
    frame_data.resize(requestedChannels.size());

    //std::cout<<"channel数量"<<requestedChannels.size() << std::endl;
    for (int i = 0; i < requestedChannels.size(); ++ i){
        frame_data.at(i).resize((width*height) *sizeof(float));
        memset(frame_data.at(i).data(), 0, sizeof((width*height) *sizeof(float)));
    }

    const int xStride = 1;
    const int yStride = width;

 //   const int xStride = height;
 //   const int yStride = 1;

    for (size_t i = 0; i != requestedChannels.size(); ++i) {
        // Allocate the memory
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
        //std::cout<<xStride <<" "<<xSampling << std::endl;
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

void readexr(std::string path, std::vector<cv::Mat> &image,
             std::vector<std::string> &channel_name) {
  InputFile img(path.c_str());
  const ChannelList &channels = img.header().channels();
  std::vector<std::string> &channelNames = channel_name;

  // Prepare the framebuffer
  const Box2i &dw = img.header().dataWindow();
  const ChannelList &imgChannels = img.header().channels();

  //获取所有的channel的名字，并返回
  getChannelNames(channels, channelNames);
  // for (int i = 0; i < channelNames.size(); ++ i){
  //    std::cout<< channelNames.at(i)<< std::endl;
  //}

  //设置对应的buffer,设置好
  FrameBuffer framebuffer;
  prepareFrameBuffer(framebuffer, dw, imgChannels, channelNames);
  //std::cout << channelNames.size() << std::endl;
  image.resize(channelNames.size());
  // Actually read the pixels

  // std::cout<<"@"<<std::endl;
  img.setFrameBuffer(framebuffer);  //设置好，

  // std::cout<<"准备读取"<<std::endl;
  img.readPixels(dw.min.y, dw.max.y);

  // std::cout<<"读取完毕"<<std::endl;

  const int width = dw.max.x - dw.min.x + 1;
  const int height = dw.max.y - dw.min.y + 1;

  for (int i = 0; i < channelNames.size(); ++i) {
    // std::cout<<"@" << i << std::endl;
    image.at(i) = cv::Mat(height, width, CV_32F);
    memset(image.at(i).data, 0, sizeof((height * width) * sizeof(float)));
    // std::cout<<"@@aa" << std::endl;

    // std::cout<<"@@bb" << std::endl;
    // std::cout<<"数据开始转移到cv mat格式上"<<std::endl;
    memmove(image.at(i).data, frame_data.at(i).data(),
            (height * width) * sizeof(float));

    // std::cout<<"@@cc" << std::endl;
    // std::cout<<"转移成功，显示图片"<<std::endl;
    // cv::imshow("abc", image.at(i));
    // cv::waitKey(-1);
  }
}
