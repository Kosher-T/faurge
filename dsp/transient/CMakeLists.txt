cmake_minimum_required(VERSION 3.10)
project(faurge-transient CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)

include_directories(include)
include_directories(${SNDFILE_INCLUDE_DIRS})

set(LIB_SOURCES
    src/transient.cpp
    src/transient_splitter.cpp
    src/transient_metrics.cpp
)

add_executable(faurge-transient src/main.cpp ${LIB_SOURCES})
target_link_libraries(faurge-transient PRIVATE ${SNDFILE_LIBRARIES})

enable_testing()

foreach(TEST passthrough attack silence)
    add_executable(test_${TEST} test/test_${TEST}.cpp ${LIB_SOURCES})
    target_link_libraries(test_${TEST} PRIVATE ${SNDFILE_LIBRARIES} m)
    add_test(NAME test_${TEST} COMMAND test_${TEST})
endforeach()
