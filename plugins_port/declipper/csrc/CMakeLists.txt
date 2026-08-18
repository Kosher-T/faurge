cmake_minimum_required(VERSION 3.14)
project(faurge_declipper_portable CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(_faurge_declip_cpp
    bindings.cpp
    src/declipper.cpp
    src/clip_detector.cpp
    src/reconstructor.cpp
    src/post_filter.cpp
    src/metrics.cpp
)

target_include_directories(_faurge_declip_cpp PRIVATE include)
target_link_libraries(_faurge_declip_cpp PRIVATE pybind11::headers)
