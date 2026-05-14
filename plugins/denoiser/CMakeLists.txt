cmake_minimum_required(VERSION 3.10)
project(faurge_denoiser CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find sndfile library
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)

# ── Model download ────────────────────────────────────────────
set(MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models")
set(MODEL_TAR "${MODEL_DIR}/DeepFilterNet3.tar.gz")
set(MODEL_EXTRACT_DIR "${MODEL_DIR}/.extracted")

if(NOT EXISTS "${MODEL_TAR}")
    message(STATUS "Downloading DeepFilterNet 3 model weights...")
    file(DOWNLOAD
        "https://github.com/Rikorose/DeepFilterNet/releases/download/v0.3.0/DeepFilterNet3.tar.gz"
        "${MODEL_TAR}"
        SHOW_PROGRESS
        STATUS model_status)
    list(GET model_status 0 model_code)
    if(model_code EQUAL 0)
        message(STATUS "Model weights downloaded: ${MODEL_TAR}")
    else()
        message(WARNING "Failed to download model weights. Build will continue but model must be provided manually.")
    endif()
endif()

# ── Rust bridge build ─────────────────────────────────────────
set(RUST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rust")
set(RUST_TARGET_DIR "${RUST_DIR}/target")
set(RUST_LIB "${RUST_TARGET_DIR}/release/libdf_bridge.a")

find_program(CARGO NAMES cargo)
if(NOT CARGO)
    message(FATAL_ERROR "Rust toolchain (cargo) not found. Install via: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh")
endif()

add_custom_command(
    OUTPUT ${RUST_LIB}
    COMMAND ${CARGO} ARGS build --release
    WORKING_DIRECTORY ${RUST_DIR}
    COMMENT "Building Rust bridge (df_bridge)..."
    DEPENDS ${RUST_DIR}/Cargo.toml ${RUST_DIR}/src/lib.rs
)

add_custom_target(rust_bridge ALL DEPENDS ${RUST_LIB})

add_library(df_bridge STATIC IMPORTED)
set_target_properties(df_bridge PROPERTIES IMPORTED_LOCATION ${RUST_LIB})
add_dependencies(df_bridge rust_bridge)

# ── Include directories ───────────────────────────────────────
include_directories(include)
include_directories(${SNDFILE_INCLUDE_DIRS})

# ── Library sources ───────────────────────────────────────────
set(LIB_SOURCES
    src/denoiser.cpp
    src/noise_estimator.cpp
    src/denoise_metrics.cpp
)

# ── Main executable ───────────────────────────────────────────
add_executable(faurge-denoise src/main.cpp ${LIB_SOURCES})
target_link_libraries(faurge-denoise PRIVATE
    df_bridge
    ${SNDFILE_LIBRARIES}
    pthread dl m
)

# ── Training pair generator ──────────────────────────────────
add_executable(faurge-generate-noisy test/generate_noisy.cpp)
target_link_libraries(faurge-generate-noisy PRIVATE ${SNDFILE_LIBRARIES})

# ── Tests ─────────────────────────────────────────────────────
enable_testing()

add_executable(test_noise_estimator test/test_noise_estimator.cpp src/noise_estimator.cpp)
target_link_libraries(test_noise_estimator PRIVATE ${SNDFILE_LIBRARIES} m)
add_test(NAME test_noise_estimator COMMAND test_noise_estimator)

add_executable(test_bridge test/test_bridge.cpp ${LIB_SOURCES})
target_link_libraries(test_bridge PRIVATE
    df_bridge
    ${SNDFILE_LIBRARIES}
    pthread dl m
)
add_dependencies(test_bridge rust_bridge)
add_test(NAME test_bridge COMMAND test_bridge)

add_executable(test_pipeline test/test_pipeline.cpp ${LIB_SOURCES})
target_link_libraries(test_pipeline PRIVATE
    df_bridge
    ${SNDFILE_LIBRARIES}
    pthread dl m
)
add_dependencies(test_pipeline rust_bridge)
add_test(NAME test_pipeline COMMAND test_pipeline)

add_executable(test_resampling test/test_resampling.cpp ${LIB_SOURCES})
target_link_libraries(test_resampling PRIVATE
    df_bridge
    ${SNDFILE_LIBRARIES}
    pthread dl m
)
add_dependencies(test_resampling rust_bridge)
add_test(NAME test_resampling COMMAND test_resampling)

add_executable(test_attenuation test/test_attenuation.cpp ${LIB_SOURCES})
target_link_libraries(test_attenuation PRIVATE
    df_bridge
    ${SNDFILE_LIBRARIES}
    pthread dl m
)
add_dependencies(test_attenuation rust_bridge)
add_test(NAME test_attenuation COMMAND test_attenuation)
