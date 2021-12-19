cmake_minimum_required(VERSION 2.8.3)
project(tftn_cmake_example)



add_definitions(-std=c++17)
find_package(OpenCV REQUIRED)
find_package(TFTN REQUIRED)

add_definitions(-DINPUT_FILE="${CMAKE_SOURCE_DIR}")


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
set_property(TARGET demo PROPERTY CXX_STANDARD 17)

message("TFTN libs:" ${TFTN_LIBS})

target_link_libraries(demo
        ${TFTN_LIBS}
        ${OpenCV_LIBS}
        -lIlmImf        #OpenEXR用的
        #-lpcgImageIO    #也是OpenEXR好像用的
        )

    return()
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
