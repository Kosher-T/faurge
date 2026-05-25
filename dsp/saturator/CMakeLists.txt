cmake_minimum_required(VERSION 3.10)
project(faurge-saturator CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)

include_directories(include)
include_directories(${SNDFILE_INCLUDE_DIRS})

set(LIB_SOURCES
    src/saturator.cpp
    src/waveshaper.cpp
    src/biquad.cpp
    src/sat_metrics.cpp
)

add_executable(faurge-saturate src/main.cpp ${LIB_SOURCES})
target_link_libraries(faurge-saturate PRIVATE ${SNDFILE_LIBRARIES})

enable_testing()

add_executable(test_drive_zero test/test_drive_zero.cpp ${LIB_SOURCES})
target_link_libraries(test_drive_zero PRIVATE ${SNDFILE_LIBRARIES} m)
add_test(NAME test_drive_zero COMMAND test_drive_zero)

add_executable(test_types test/test_types.cpp ${LIB_SOURCES})
target_link_libraries(test_types PRIVATE ${SNDFILE_LIBRARIES} m)
add_test(NAME test_types COMMAND test_types)

add_executable(test_hpf test/test_hpf.cpp ${LIB_SOURCES})
target_link_libraries(test_hpf PRIVATE ${SNDFILE_LIBRARIES} m)
add_test(NAME test_hpf COMMAND test_hpf)
