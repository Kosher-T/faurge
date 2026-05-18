cmake_minimum_required(VERSION 3.10)
project(faurge_exciter CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)

include_directories(include)
include_directories(${SNDFILE_INCLUDE_DIRS})

set(LIB_SOURCES
    src/crossover_filter.cpp
    src/high_band.cpp
    src/low_band.cpp
    src/exciter.cpp
    src/exciter_metrics.cpp
)

add_executable(faurge-excite src/main.cpp ${LIB_SOURCES})
target_link_libraries(faurge-excite PRIVATE ${SNDFILE_LIBRARIES})

add_executable(faurge-generate-excite-pairs test/generate_pairs.cpp)
target_link_libraries(faurge-generate-excite-pairs PRIVATE ${SNDFILE_LIBRARIES})

enable_testing()

add_executable(test_high_band test/test_high_band.cpp ${LIB_SOURCES})
target_link_libraries(test_high_band PRIVATE ${SNDFILE_LIBRARIES} m)
add_test(NAME test_high_band COMMAND test_high_band)

add_executable(test_low_band test/test_low_band.cpp ${LIB_SOURCES})
target_link_libraries(test_low_band PRIVATE ${SNDFILE_LIBRARIES} m)
add_test(NAME test_low_band COMMAND test_low_band)

add_executable(test_pipeline test/test_pipeline.cpp ${LIB_SOURCES})
target_link_libraries(test_pipeline PRIVATE ${SNDFILE_LIBRARIES} m)
add_test(NAME test_pipeline COMMAND test_pipeline)

add_executable(test_metrics test/test_metrics.cpp ${LIB_SOURCES})
target_link_libraries(test_metrics PRIVATE ${SNDFILE_LIBRARIES} m)
add_test(NAME test_metrics COMMAND test_metrics)
