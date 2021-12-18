cmake_minimum_required(VERSION 2.8.3)
project(tftn_cmake_example)



#you can change this version
#why I use opencv3.3? because my OS is ubuntu18.04,
#and it default opencv is opencv3.2.  the default opencv has no openexr,
#so we need to install openexr and reinstall opencv by ourselves.
#In this situtaion, we download opencv3.3, rebuild it and install it to /opt/opencv3.3
find_package(OpenCV 3.2 REQUIRED EXACT)
find_package(TFTN REQUIRED)

add_definitions(-DINPUT_FILE="${CMAKE_SOURCE_DIR}")
#add_definitions(-Dthrow\(...\))


include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${TFTN_INCLUDE_DIRS}
)

add_executable(demo
        ${CMAKE_SOURCE_DIR}/demo.cpp
#        #include/tftn/tftn.cpp
#        ThirdParty/cvrgbd/src/depth_to_3d.cpp
#        ThirdParty/cvrgbd/src/utils.cpp
        )
#very important!!
#otherwise in some machine,it will be built with c++11, you will find some error!!!!
#openexr can only be used in CXX14
set_property(TARGET demo PROPERTY CXX_STANDARD 14)


target_link_libraries(demo
        ${TFTN_LIBS}
        ${OpenCV_LIBS}
        -lIlmImf        #OpenEXR用的
        #-lpcgImageIO    #也是OpenEXR好像用的
        )

