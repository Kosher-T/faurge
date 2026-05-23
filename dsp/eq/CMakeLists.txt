cmake_minimum_required(VERSION 3.10)
project(faurge-eq CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)

include_directories(include)
include_directories(${SNDFILE_INCLUDE_DIRS})

set(LIB_SOURCES
    src/biquad.cpp
    src/eq.cpp
    src/eq_metrics.cpp
)

add_executable(faurge-eq src/main.cpp ${LIB_SOURCES})
target_link_libraries(faurge-eq PRIVATE ${SNDFILE_LIBRARIES})

enable_testing()

add_executable(test_sweep test/test_sweep.cpp ${LIB_SOURCES})
target_link_libraries(test_sweep PRIVATE ${SNDFILE_LIBRARIES} m)
add_test(NAME test_sweep COMMAND test_sweep)

add_executable(test_notch test/test_notch.cpp ${LIB_SOURCES})
target_link_libraries(test_notch PRIVATE ${SNDFILE_LIBRARIES} m)
add_test(NAME test_notch COMMAND test_notch)

add_executable(test_skew test/test_skew.cpp ${LIB_SOURCES})
target_link_libraries(test_skew PRIVATE ${SNDFILE_LIBRARIES} m)
add_test(NAME test_skew COMMAND test_skew)

add_executable(test_dynamic test/test_dynamic.cpp ${LIB_SOURCES})
target_link_libraries(test_dynamic PRIVATE ${SNDFILE_LIBRARIES} m)
add_test(NAME test_dynamic COMMAND test_dynamic)
