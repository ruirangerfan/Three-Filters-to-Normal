cmake_minimum_required(VERSION 2.8.3)
project(tftn_cmake_example)



add_definitions(-std=c++17)
find_package(OpenCV  REQUIRED)
find_package(TFTN REQUIRED)
find_package(PythonLibs REQUIRED) #默认支持python3,不想支持2

add_definitions(-DINPUT_FILE="${CMAKE_SOURCE_DIR}")

#dont need here, just need to reboot your system, or run <sudo ldconfig> 
link_directories(
    #TODO
    #/usr/local/lib/
)




include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${TFTN_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIR}
)

add_executable(demo
        ${CMAKE_SOURCE_DIR}/demo.cpp
        )
#very important!!
#otherwise in some machine,it will be built with c++11, you will find some error!!!!
#openexr can only be used in CXX14
set_property(TARGET demo PROPERTY CXX_STANDARD 17)

message("TFTN libs:" ${TFTN_LIBS})

target_link_libraries(demo
        ${TFTN_LIBS}
        ${OpenCV_LIBS}
        #-lIlmImf        #OpenEXR用的
        #-lpcgImageIO    #也是OpenEXR好像用的
        )


add_executable(loadpy
        ${CMAKE_SOURCE_DIR}/loadpy.cpp
        )
target_link_libraries(loadpy
        ${TFTN_LIBS}
        ${OpenCV_LIBS}
        ${PYTHON_LIBRARY}
        )





return()
get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()
