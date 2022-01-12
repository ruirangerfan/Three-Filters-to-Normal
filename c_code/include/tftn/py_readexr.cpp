//
// Created by zpmc on 2021/12/18.
//


//sudo apt-get install pybind11-dev 安装pybind11

#ifdef USE_PYTHON
//#define PY_SSIZE_T_CLEAN //Make "s#" use Py_ssize_t rather than int
#include <Python.h> //这个一定要放在引用其他东西的前面，因为他可能重写一些标准的函数

#include <pybind11/pybind11.h>

#include <numpy/arrayobject.h>

#include <iostream>

//#include "tftn/py_readexr.h"
#include "tftn/readexr.h" //这行




//https://docs.python.org/3.8/extending/extending.htm
//static PyObject* ReadEXRPY(char* input_path="asdf") {
    static int ReadEXRPY(char* input_path="asdf") {
    //PyObject *self;

    Py_Initialize();
    _import_array();  //先运行 始化 numpy
    //执行环境，主要是导入包，python2.7用void返回类型，python3.0以上用int返回类型

    PyObject *ArgArray =
            PyTuple_New(2);  //返回类型是2个参数，一个是numpy，一个是字母对应的xyz

    std::vector<cv::Mat> cv_image;
    std::vector<std::string> channel_name;
    readexr(input_path, cv_image, channel_name);
    // TODO 检查文件路径是否合法

    int rows = cv_image.at(0).rows;
    int cols = cv_image.at(0).cols;

    std::vector<float> data;
    data.resize(rows * cols * channel_name.size());
    for (int i = 0; i < channel_name.size(); ++i) {
        memmove(data.data() + i * rows * cols, cv_image.at(i).data,
                rows * cols * sizeof(float));
    }

    npy_intp dims[3] = {rows, cols, static_cast<npy_intp>(channel_name.size())};  //给定维度信息

    PyObject *PyArray = PyArray_SimpleNewFromData(
            3, dims, NPY_FLOAT,
            data.data());  //生成包含这个多维数组的PyObject对象，使用PyArray_SimpleNewFromData函数，第一个参数2表示维度，第二个为维度数组Dims,第三个参数指出数组的类型，第四个参数为数组
    //回头把PyArray传递出去
    // data这里是否是指针？还是静态数组也可以

    Py_Finalize();
    //return PyArray;
    //测试结果
}
#endif
