
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "tftn/readexr.h"
namespace py = pybind11;


//input :
py::array_t<double> add_arrays(py::array_t<double> input1,
                               py::array_t<double> input2) {
  py::buffer_info buf1 = input1.request(), buf2 = input2.request();

  if (buf1.ndim != 1 || buf2.ndim != 1)
    throw std::runtime_error("Number of dimensions must be one");

  if (buf1.size != buf2.size)
    throw std::runtime_error("Input shapes must match");

  /* No pointer is passed, so NumPy will allocate the buffer */
  auto result = py::array_t<double>(buf1.size);

  py::buffer_info buf3 = result.request();

  double *ptr1 = static_cast<double *>(buf1.ptr);
  double *ptr2 = static_cast<double *>(buf2.ptr);
  double *ptr3 = static_cast<double *>(buf3.ptr);

  for (size_t idx = 0; idx < buf1.shape[0]; idx++)
    ptr3[idx] = ptr1[idx] + ptr2[idx];

  return result;
}



//more information can be found here: 
//https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html
py::array_t<double> readexr_py(const std::string &path) {
  std::vector<cv::Mat> image;
  std::vector<std::string> channel_name;

  //读取exr格式的数据，这里是默认返回opencv格式
  readexr(path, image, channel_name);

  uint32_t channel_size = channel_name.size();
  uint32_t rows = image.at(0).rows;
  uint32_t cols = image.at(0).cols;

  py::array_t<double> result({channel_size, rows, cols});
  py::buffer_info buf = result.request();
  double *ptr = static_cast<double *>(buf.ptr);

  //把image数据格式改为double类型,好像python没有真正的float
  py::array_t<double, py::array::c_style | py::array::forcecast> result_numpy;
  for (int i = 0; i < image.size(); ++i) {
    cv::Mat tmp;
    image.at(i).convertTo(tmp, CV_64FC(channel_name.size()));
    image.at(i) = tmp;
  }
 

    //数据转移
  for (int i = 0; i < channel_name.size(); ++i) {
    memmove(ptr + i * rows * cols, image.at(i).data, rows * cols * sizeof(double));
  }
  return result;


  /*
  result.shape({channel_size, rows, cols});
  py::buffer_info buffer;


  //先尝试只返回一个image

  //数据类型有的c_style是行保存的矩阵风格
 */

  // py::buffer_info buf1 = input1.path(), buf2 = input2.request();
  // path.request()
}
PYBIND11_PLUGIN(readexr_py) {
  py::module m("readexr_py", "EXR format IO");

  m.def("readexr_py", &readexr_py, "can be used to read a exr format file");
//  m.def("pcc_encoder", &pcc_encoder, "Encoder the pointcloud data");
 // m.def("pcc_decoder", &pcc_decoder, "Decoder the pointcloud data");

  return m.ptr();
}

/*
pybind11::class_<cv::Mat>(data, "cvMat", pybind11::buffer_protocol())
    .def(pybind11::init([](pybind11::buffer b) {
      pybind11::buffer_info info = b.request();
      if (info.format != pybind11::format_descriptor<float>::format())
        throw std::runtime_error("数据格式不匹配。这里应该是float类型\n");
      if (info.ndim != 2) throw std::runtime_error("维度不正确\n");
      //....就是从python读了一个b，然后返回一个matrix格式.
      return cv::Mat();
    }));

pybind11::class_<cv::Mat>(data, "cvMat", pybind11::buffer_protocol())
    .def_buffer([](cv::Mat &data) {
      int cols = data.cols;
      int rows = data.rows;
      pybind11::buffer_info buff;
      buff.buf = data.data;  //数据本身
      // data.step
      buff.itemsize = sizeof(float);  //每个size大小
      buff.format = pybind11::format_descriptor<float>::format();
      buff.ndim = 3;                 //三维数组
      buff.shape = {rows, cols, 3};  // N维
      buff.strides = {sizeof(float) * cols * rows, sizeof(float) * rows,
                      sizeof(float)};  //感觉应该是这个意思.维度从高到低
    });
    */

// https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html

//测试给出返回2个值
std::string pcc_encoder() {
  printf("@@\n");
  return "asf";
}

int pcc_decoder() {
  printf("!!\n");
  return 123;
}

void foo() {
  cv::Mat data;
  //从opencv读过来的numpy格式，自动转为cvmat
  //从 C++的cvmat，传输给python的numpy
}


