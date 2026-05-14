#include "faurge/denoiser.hpp"
#include "faurge/noise_estimator.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define TEST(name) \
    static void test_##name(); \
    struct Register_##name { \
        Register_##name() { tests.push_back({#name, test_##name}); } \
    } reg_##name; \
    static void test_##name()

struct TestEntry { const char* name; void (*fn)(); };
static std::vector<TestEntry> tests;
static int failures = 0;

#define ASSERT_TRUE(x) do { \
    if (!(x)) { \
        fprintf(stderr, "  FAIL: %s (line %d)\n", #x, __LINE__); \
        ++failures; \
        return; \
    } \
} while(0)

extern "C" {
    void* df_bridge_create(const char* model_path, float atten_limit);
    int   df_bridge_process(void* handle, float* samples, size_t num_samples);
    size_t df_bridge_get_frame_size(void* handle);
    size_t df_bridge_get_sample_rate(void* handle);
    void  df_bridge_destroy(void* handle);
}

TEST(bridge_create_destroy) {
    void* handle = df_bridge_create(nullptr, 0.78f);
    ASSERT_TRUE(handle != nullptr);
    df_bridge_destroy(handle);
    fprintf(stderr, "    create/destroy: OK\n");
}

TEST(bridge_returns_48k_sample_rate) {
    void* handle = df_bridge_create(nullptr, 0.78f);
    ASSERT_TRUE(handle != nullptr);
    size_t sr = df_bridge_get_sample_rate(handle);
    ASSERT_TRUE(sr == 48000);
    fprintf(stderr, "    Sample rate: %zu Hz\n", sr);
    df_bridge_destroy(handle);
}

TEST(bridge_frame_size_is_positive) {
    void* handle = df_bridge_create(nullptr, 0.78f);
    ASSERT_TRUE(handle != nullptr);
    size_t fs = df_bridge_get_frame_size(handle);
    ASSERT_TRUE(fs > 0);
    fprintf(stderr, "    Frame size: %zu\n", fs);
    df_bridge_destroy(handle);
}

TEST(bridge_processes_silence_without_crash) {
    void* handle = df_bridge_create(nullptr, 0.78f);
    ASSERT_TRUE(handle != nullptr);

    size_t fs = df_bridge_get_frame_size(handle);
    std::vector<float> buf(fs, 0.0f);

    int ret = df_bridge_process(handle, buf.data(), buf.size());
    ASSERT_TRUE(ret == 0);
    df_bridge_destroy(handle);
    fprintf(stderr, "    Silence round-trip: OK\n");
}

TEST(bridge_processes_sine_without_clipping) {
    void* handle = df_bridge_create(nullptr, 0.78f);
    ASSERT_TRUE(handle != nullptr);

    size_t fs = df_bridge_get_frame_size(handle);
    std::vector<float> buf(fs);
    for (size_t i = 0; i < fs; ++i)
        buf[i] = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f * i / 48000.0f);

    int ret = df_bridge_process(handle, buf.data(), buf.size());
    ASSERT_TRUE(ret == 0);

    for (size_t i = 0; i < fs; ++i) {
        ASSERT_TRUE(buf[i] >= -1.0f);
        ASSERT_TRUE(buf[i] <= 1.0f);
    }

    df_bridge_destroy(handle);
    fprintf(stderr, "    Sine in [-1, 1] range: OK\n");
}

int main() {
    int passed = 0, failed = 0;
    fprintf(stderr, "\n=== Denoiser: Bridge Lifecycle Tests ===\n\n");
    for (const auto& t : tests) {
        failures = 0;
        fprintf(stderr, "  [RUN]  %s\n", t.name);
        t.fn();
        if (failures == 0) {
            fprintf(stderr, "  [PASS] %s\n", t.name);
            ++passed;
        } else {
            fprintf(stderr, "  [FAIL] %s\n", t.name);
            ++failed;
        }
    }
    fprintf(stderr, "\n  Results: %d passed, %d failed\n\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
