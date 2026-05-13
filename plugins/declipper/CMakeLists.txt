cmake_minimum_required(VERSION 3.10)
project(faurge_declipper CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find sndfile library
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)

include_directories(include)
include_directories(${SNDFILE_INCLUDE_DIRS})

# Library sources
set(LIB_SOURCES
    src/clip_detector.cpp
    src/declipper.cpp
    src/metrics.cpp
    src/post_filter.cpp
    src/reconstructor.cpp
)

# Main executable
add_executable(faurge-declip src/main.cpp ${LIB_SOURCES})
target_link_libraries(faurge-declip PRIVATE ${SNDFILE_LIBRARIES})

# Training pair generator
add_executable(faurge-generate-clipped test/generate_clipped.cpp)
target_link_libraries(faurge-generate-clipped PRIVATE ${SNDFILE_LIBRARIES})

# Tests
enable_testing()

add_executable(test_detector test/test_detector.cpp ${LIB_SOURCES})
target_link_libraries(test_detector PRIVATE ${SNDFILE_LIBRARIES})
add_test(NAME test_detector COMMAND test_detector)

add_executable(test_pipeline test/test_pipeline.cpp ${LIB_SOURCES})
target_link_libraries(test_pipeline PRIVATE ${SNDFILE_LIBRARIES})
add_test(NAME test_pipeline COMMAND test_pipeline)

add_executable(test_reconstructor test/test_reconstructor.cpp ${LIB_SOURCES})
target_link_libraries(test_reconstructor PRIVATE ${SNDFILE_LIBRARIES})
add_test(NAME test_reconstructor COMMAND test_reconstructor)
