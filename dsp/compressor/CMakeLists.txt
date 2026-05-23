cmake_minimum_required(VERSION 3.10)
project(faurge-compressor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)

include_directories(include)
include_directories(${SNDFILE_INCLUDE_DIRS})

set(LIB_SOURCES
    src/compressor.cpp
    src/envelope.cpp
    src/gain_computer.cpp
    src/comp_metrics.cpp
)

add_executable(faurge-compress src/main.cpp ${LIB_SOURCES})
target_link_libraries(faurge-compress PRIVATE ${SNDFILE_LIBRARIES})

enable_testing()

add_executable(test_passthrough test/test_passthrough.cpp ${LIB_SOURCES})
target_link_libraries(test_passthrough PRIVATE ${SNDFILE_LIBRARIES} m)
add_test(NAME test_passthrough COMMAND test_passthrough)

add_executable(test_pumping test/test_pumping.cpp ${LIB_SOURCES})
target_link_libraries(test_pumping PRIVATE ${SNDFILE_LIBRARIES} m)
add_test(NAME test_pumping COMMAND test_pumping)

add_executable(test_lookahead test/test_lookahead.cpp ${LIB_SOURCES})
target_link_libraries(test_lookahead PRIVATE ${SNDFILE_LIBRARIES} m)
add_test(NAME test_lookahead COMMAND test_lookahead)

add_executable(test_detector test/test_detector.cpp ${LIB_SOURCES})
target_link_libraries(test_detector PRIVATE ${SNDFILE_LIBRARIES} m)
add_test(NAME test_detector COMMAND test_detector)

add_executable(test_sidechain test/test_sidechain.cpp ${LIB_SOURCES})
target_link_libraries(test_sidechain PRIVATE ${SNDFILE_LIBRARIES} m)
add_test(NAME test_sidechain COMMAND test_sidechain)

add_executable(test_stereo_link test/test_stereo_link.cpp ${LIB_SOURCES})
target_link_libraries(test_stereo_link PRIVATE ${SNDFILE_LIBRARIES} m)
add_test(NAME test_stereo_link COMMAND test_stereo_link)
