//
// Created by zpmc on 2021/12/18.
//

#ifndef TFTN_PY_READEXR_H
#define TFTN_PY_READEXR_H



#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>

#include <Python.h>
#include <numpy/arrayobject.h>

extern "C" {
PyObject *ReadEXRPY(char* input_path);
}




#endif //TFTN_PY_READEXR_H
