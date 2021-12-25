#include <tftn/py_readexr.h>

std::string input_file = std::string(INPUT_FILE) + "/0001.exr" ;
int main() {

  PyObject* x = ReadEXRPY(input_file);
  return 0;
}