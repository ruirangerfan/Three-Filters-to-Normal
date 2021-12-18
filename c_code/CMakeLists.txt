cmake_minimum_required(VERSION 2.8.3)
project(tftn)


string(ASCII 27 Esc)
set(ColourReset "${Esc}[m")
set(ColourBold  "${Esc}[1m")
set(Red         "${Esc}[31m")
set(Green       "${Esc}[32m")
set(Yellow      "${Esc}[33m")
set(Blue        "${Esc}[34m")
set(Magenta     "${Esc}[35m")
set(Cyan        "${Esc}[36m")
set(White       "${Esc}[37m")
set(BoldRed     "${Esc}[1;31m")
set(BoldGreen   "${Esc}[1;32m")
set(BoldYellow  "${Esc}[1;33m")
set(BoldBlue    "${Esc}[1;34m")
set(BoldMagenta "${Esc}[1;35m")
set(BoldCyan    "${Esc}[1;36m")
set(BoldWhite   "${Esc}[1;37m")


#you can change this version
find_package(OpenCV  REQUIRED)
#find_package(Ceres REQUIRED)
#find_package(Eigen3 REQUIRED)


set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mfma -msse2 -msse3 -msse4")
message("${Green}install path: ${CMAKE_INSTALL_PREFIX}${ColourReset}")
#set(CMAKE_INSTALL_PREFIX "/opt/tftn")



add_definitions(-std=c++17)
add_definitions(-mfma)
add_definitions(-mno-avx512f)
add_definitions(-mno-avx512vl)
add_definitions(-DUSE_SSE_AUTO)
add_definitions(-DSSE_MATHFUN_WITH_CODE)
set(CPU_ALL_OPTIMIZATIONS "SSE;SSE2;SSE3;SSSE3;SSE4_1;SSE4_2;POPCNT;AVX;FP16;AVX2;FMA3;AVX_512F;AVX512VL;AVX512F;AVX_512VL")
#SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -mavx512f -mfma -mavx512vl -mavx512bw -mavx512dq -march=knl -march=sandybridge -march=haswell")
#message('0-0000000000000000000' ${AVX_FLAGS})

#ADD_DEFINITIONS(-DEIGEN_ENABLE_AVX512=ON) #maybe we dont need eigen, i fortget it.

#create static .a file.
# 一个不错的cmake教程 https://blog.csdn.net/u012564117/article/details/95085360
# 另一个不错的cmake教程 https://zhuanlan.zhihu.com/p/87738686
file(GLOB ALL_HEADERS include/tftn/*.h)
file(GLOB ALL_SOURCES include/*/*.cpp)
#aux_source_directory(include/tftfn/ ALL_SOURCES )

message("${BoldBlue}ALL_HEADERS: ${ColourReset}")
message(${ALL_HEADERS})
message(${BoldBlue}ALL_SOURCES: ${ColourReset})
message(${ALL_SOURCES})



#add_library(tftn_static STATIC include/tftn/tftn.cpp)
add_library(tftn_static STATIC include/tftn/tftn.cpp)
add_library(tftn_dynamic SHARED include/tftn/tftn.cpp include/tftn/readexr.cpp include/tftn/readexr.h)

#set(LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
#set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
SET_TARGET_PROPERTIES(tftn_static PROPERTIES OUTPUT_NAME "tftn") #modify the static file name
SET_TARGET_PROPERTIES(tftn_dynamic PROPERTIES OUTPUT_NAME "tftn") #modify the static file name


include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_NAME}
        include
        ThirdParty
)

#add_executable(example
#        example/demo.cpp
#        #include/tftn/tftn.cpp
#        ThirdParty/cvrgbd/src/depth_to_3d.cpp
#        ThirdParty/cvrgbd/src/utils.cpp
#        )

#very important!!
#otherwise in some machine,it will be built with c++11, you will find some error!!!!
#set_property(TARGET example PROPERTY CXX_STANDARD 17)


#target_link_libraries(example
#        ${OpenCV_LIBS}
#        tftn
#        )
#

# install libtftn.so and libtftn_static.a
#install(
#        TARGETS tftn tftn_static
#        LIBRARY DESTINATION lib/${PROJECT_NAME}  #静态库
#        ARCHIVE DESTINATION lib/${PROJECT_NAME}
#        EXPORT  DESTINATION lib/${PROJECT_NAME}
#) #动态库

# https://blog.csdn.net/qq_38410730/article/details/102837401
set(CMAKE_INSTALL_LIBDIR  ${CMAKE_INSTALL_PREFIX}/lib)
set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX}/include)

message( "${BoldBlue}install library dir:"${ColourReset}\n ${CMAKE_INSTALL_LIBDIR})
message( "${BoldBlue}install include dir:"${ColourReset}\n ${CMAKE_INSTALL_INCLUDEDIR}	)

#export(EXPORT tftn NAMESPACE tftn:: FILE hahaha.cmake)
#主要用于生成一个 TFTNTargets.cmake，用来给其他目录做子目录时候用。教程https://blog.csdn.net/weixin_30379531/article/details/98589931
export(TARGETS tftn_static #tftn_dynamic
        APPEND FILE "${CMAKE_BINARY_DIR}/TFTNTargets.cmake"
        )

install(
        TARGETS tftn_static #tftn_dynamic
        DESTINATION lib
        EXPORT ${PROJECT_NAME} #不知道为啥这里只能是这个project_name
) #安装静态链接库

#install(FILES ${PROJECT_SOURCE_DIR}/include/tftn/tftn.h DESTINATION include)
#安装头文件
install(FILES ${ALL_HEADERS} DESTINATION include/${PROJECT_NAME})

#安装tftn.cmake tftn-release.cmake
install(EXPORT ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/tftn)



#配置动态库
configure_file(${CMAKE_SOURCE_DIR}/cmake/TFTNConfig.cmake.in ${PROJECT_BINARY_DIR}/TFTNConfig.cmake @ONLY)
install(FILES ${PROJECT_BINARY_DIR}/TFTNConfig.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tftn)

message("!!"${CMAKE_CURRENT_LIST_FILE})


#set(CMAKE_MODULE_PATH ${CMAKE_ROOT}/Modules)
# https://zhuanlan.zhihu.com/p/422919663 CMake编写install export