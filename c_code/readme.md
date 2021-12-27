
then you can use it.
an CMAKE example can be see in 'CmakeExample'

and you can easily to use our code to load EXR format file, and convert it to opencv cv::Mat


写个中文的说明先。

回头再改。

# 基本依赖
```
sudo apt-get install openexr
```


# C++ 教程

```
 mkdir build
 cmake ..
 sudo make install
```
随后可以参考目录下CmakeExample的结构，构造一个cmake项目，可以方便的使用本代码。


# Python 教程

目前python只实现了读取openexr格式的读取操作,

首先确保电脑已经安装python. ubuntu 18.04默认是python2.7和python3.6
ubuntu 20.04 默认的python版本是python3.8
当然你可以用apt-get install libpython3.8 或者 apt-get install libpython3.7 之类的来安装你想要的版本
当然，别忘了使用sudo apt-get install python3.8-dev 或者 sudo apt-get install python3.7-dev 来安装相关的头文件

```
mkdir build
cmake ..
sudo make install
```

此时文件夹里，应该出现一个新的文件夹叫 python_module_output
里面有一个readexr_py或者readexr_py.so的文件。
不管生成了啥，重命名为readexr_py.so即可。
这是一个宝贝文件，他是可以被import的。
你把他放入比如/usr/local/lib/python3.6/dist-packages/里面，新建一个文件夹就叫readexr_py,然后把他放在文件夹里就可以了。
注意！！！这里python3.6或者python3.7，取决于你cmake的时候输出的的cmake版本。
如果想指定python的版本的话，在CMakeLists.txt中修改PythonLibs后面的版本号即可。

这时候，你可以运行PythonExample文件夹里的程序demo.py
这是一个示范如何读取exr格式的程序。【因为我还没有制作让python有多个返回值。所以先将就一下，我歇歇再写】






