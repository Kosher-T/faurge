cmake_minimum_required(VERSION 3.10)
project(faurge-limiter CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)

include_directories(include)
include_directories(${SNDFILE_INCLUDE_DIRS})

set(LIB_SOURCES
    src/limiter.cpp
    src/peak_predictor.cpp
    src/oversampler.cpp
    src/biquad.cpp
    src/limiter_metrics.cpp
)

add_executable(faurge-limit src/main.cpp ${LIB_SOURCES})
target_link_libraries(faurge-limit PRIVATE ${SNDFILE_LIBRARIES})

enable_testing()

add_executable(test_soft_limit test/test_soft_limit.cpp ${LIB_SOURCES})
target_link_libraries(test_soft_limit PRIVATE ${SNDFILE_LIBRARIES} m)
add_test(NAME test_soft_limit COMMAND test_soft_limit)

add_executable(test_hard_limit test/test_hard_limit.cpp ${LIB_SOURCES})
target_link_libraries(test_hard_limit PRIVATE ${SNDFILE_LIBRARIES} m)
add_test(NAME test_hard_limit COMMAND test_hard_limit)

add_executable(test_ceiling test/test_ceiling.cpp ${LIB_SOURCES})
target_link_libraries(test_ceiling PRIVATE ${SNDFILE_LIBRARIES} m)
add_test(NAME test_ceiling COMMAND test_ceiling)

add_executable(test_lookahead test/test_lookahead.cpp ${LIB_SOURCES})
target_link_libraries(test_lookahead PRIVATE ${SNDFILE_LIBRARIES} m)
add_test(NAME test_lookahead COMMAND test_lookahead)

add_executable(test_stereo_link test/test_stereo_link.cpp ${LIB_SOURCES})
target_link_libraries(test_stereo_link PRIVATE ${SNDFILE_LIBRARIES} m)
add_test(NAME test_stereo_link COMMAND test_stereo_link)
