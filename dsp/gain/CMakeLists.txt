cmake_minimum_required(VERSION 3.10)
project(faurge-gain CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)

include_directories(include)
include_directories(${SNDFILE_INCLUDE_DIRS})

set(LIB_SOURCES
    src/gain.cpp
    src/gain_metrics.cpp
)

add_executable(faurge-gain src/main.cpp ${LIB_SOURCES})
target_link_libraries(faurge-gain PRIVATE ${SNDFILE_LIBRARIES})

enable_testing()

foreach(TEST identity balance clipping lufs)
    add_executable(test_${TEST} test/test_${TEST}.cpp ${LIB_SOURCES})
    target_link_libraries(test_${TEST} PRIVATE ${SNDFILE_LIBRARIES} m)
    add_test(NAME test_${TEST} COMMAND test_${TEST})
endforeach()
