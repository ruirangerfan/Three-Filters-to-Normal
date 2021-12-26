from distutils.core import setup
from distutils.extension import Extension

setup(name="readexr",
      ext_modules=[
          Extension("readexr", ["../include/tftn/py_readexr.cpp"],
                    libraries = ["boost_python"])
      ])