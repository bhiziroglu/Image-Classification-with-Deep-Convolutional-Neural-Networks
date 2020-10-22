cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dnn)

find_package(Torch REQUIRED)

add_executable(dnn main.cpp dataset.h dataset.cpp alexnet.h alexnet.cpp)
target_link_libraries(dnn "${TORCH_LIBRARIES}")
set_property(TARGET dnn PROPERTY CXX_STANDARD 14)