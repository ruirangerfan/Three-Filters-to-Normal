
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "tftn/readexr.h"


//测试给出返回2个值
std::string pcc_encoder(){
    printf("@@\n");
    return "asf";
}

int pcc_decoder(){
    printf("!!\n");
    return 123;
}


void foo(){

    cv::Mat data;
    //从opencv读过来的numpy格式，自动转为cvmat
    pybind11::class_<cv::Mat>(data, "cvMat", pybind11::buffer_protocol() ).def(
            pybind11::init(
                    [](pybind11::buffer b){
                        pybind11::buffer_info info = b.request();
                        if (info.format != pybind11::format_descriptor<float>::format())
                            throw std::runtime_error("数据格式不匹配。这里应该是float类型\n");
                        if (info.ndim != 2)
                            throw std::runtime_error("维度不正确\n");
                        //....就是从python读了一个b，然后返回一个matrix格式.
                        return cv::Mat();
                    }
            )
    );
    //https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html

    //从 C++的cvmat，传输给python的numpy
    pybind11::class_<cv::Mat>(data, "cvMat", pybind11::buffer_protocol()).def_buffer(
            [](cv::Mat &data){
                int cols = data.cols;
                int rows = data.rows;
                pybind11::buffer_info buff;
                buff.buf = data.data; //数据本身
                //data.step
                buff.itemsize = sizeof(float); //每个size大小
                buff.format = pybind11::format_descriptor<float>::format();
                buff.ndim = 3; //三维数组
                buff.shape={rows,cols,3}; //N维
                buff.strides = {sizeof(float)*cols * rows, sizeof(float) * rows, sizeof(float)}; //感觉应该是这个意思.维度从高到低
            }
    );





}



namespace py = pybind11;
PYBIND11_PLUGIN(Pypcc) {
    py::module m("Pypcc", "pcc python module");

    m.def("pcc_encoder", &pcc_encoder, "Encoder the pointcloud data");
    m.def("pcc_decoder", &pcc_decoder, "Decoder the pointcloud data");

    return m.ptr();
}

