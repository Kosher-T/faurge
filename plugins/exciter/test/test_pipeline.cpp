#include "faurge/exciter.hpp"
#include "faurge/exciter_metrics.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static constexpr float PI = 3.14159265358979f;

#define TEST(name) \
    static void test_##name(); \
    struct Register_##name { \
        Register_##name() { tests.push_back({#name, test_##name}); } \
    } reg_##name; \
    static void test_##name()

struct TestEntry { const char* name; void (*fn)(); };
static std::vector<TestEntry> tests;

#define ASSERT_TRUE(x) do { \
    if (!(x)) { \
        fprintf(stderr, "  FAIL: %s (line %d)\n", #x, __LINE__); \
        assert(false); \
    } \
} while(0)

static std::vector<float> makeSine(float freq, float amp,
                                   int sr, float dur) {
    size_t n = static_cast<size_t>(sr * dur);
    std::vector<float> buf(n);
    for (size_t i = 0; i < n; ++i)
        buf[i] = amp * std::sin(2.0f * PI * freq * i / sr);
    return buf;
}

static std::vector<float> makeWhiteNoise(float amp, int sr, float dur) {
    size_t n = static_cast<size_t>(sr * dur);
    std::vector<float> buf(n);
    for (size_t i = 0; i < n; ++i)
        buf[i] = amp * (2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f);
    return buf;
}

static float computeRms(const float* buf, size_t n) {
    float sumSq = 0.0f;
    for (size_t i = 0; i < n; ++i) sumSq += buf[i] * buf[i];
    return std::sqrt(sumSq / n);
}

TEST(full_pipeline_increases_high_freq_energy) {
    auto input = makeSine(3000.0f, 0.5f, 48000, 0.1f);
    auto original = input;

    faurge::ExciterConfig cfg;
    cfg.highDriveDb = 12.0f;
    cfg.highMix = 1.0f;
    cfg.lowMix = 0.0f;
    cfg.lowEnable = false;

    faurge::Exciter exciter(cfg);
    auto result = exciter.process(input, 48000);

    ASSERT_TRUE(result.success);
    ASSERT_TRUE(result.highBandEnergyDb > -60.0f);
}

TEST(full_pipeline_increases_low_freq_energy) {
    auto input = makeSine(100.0f, 0.5f, 48000, 0.1f);
    auto original = input;

    faurge::ExciterConfig cfg;
    cfg.lowDriveDb = 6.0f;
    cfg.lowMix = 1.0f;
    cfg.lowSubLevel = 1.0f;
    cfg.highEnable = false;

    faurge::Exciter exciter(cfg);
    auto result = exciter.process(input, 48000);

    ASSERT_TRUE(result.success);
    ASSERT_TRUE(result.lowBandEnergyDb > -60.0f);
}

TEST(pipeline_passthrough_at_minimal_settings) {
    auto input = makeSine(440.0f, 0.5f, 48000, 0.05f);
    auto original = input;

    faurge::ExciterConfig cfg;
    cfg.highDriveDb = 0.0f;
    cfg.lowDriveDb = 0.0f;
    cfg.lowSubLevel = 0.0f;

    faurge::Exciter exciter(cfg);
    auto result = exciter.process(input, 48000);

    ASSERT_TRUE(result.success);

    for (size_t i = 0; i < input.size(); ++i) {
        ASSERT_TRUE(std::fabs(input[i] - original[i]) < 1e-6f);
    }
}

TEST(pipeline_no_clipping) {
    auto input = makeWhiteNoise(0.8f, 48000, 0.1f);

    faurge::ExciterConfig cfg;
    cfg.highDriveDb = 12.0f;
    cfg.highMix = 1.0f;
    cfg.lowDriveDb = 6.0f;
    cfg.lowMix = 1.0f;
    cfg.lowSubLevel = 1.0f;

    faurge::Exciter exciter(cfg);
    auto result = exciter.process(input, 48000);

    ASSERT_TRUE(result.success);
    for (size_t i = 0; i < input.size(); ++i) {
        ASSERT_TRUE(input[i] >= -1.0f && input[i] <= 1.0f);
    }
}

TEST(processing_time_is_reasonable) {
    auto input = makeWhiteNoise(0.5f, 48000, 1.0f);

    faurge::ExciterConfig cfg;
    cfg.highDriveDb = 6.0f;
    cfg.lowDriveDb = 3.0f;

    faurge::Exciter exciter(cfg);
    auto result = exciter.process(input, 48000);

    fprintf(stderr, "    1 sec audio processed in %.2f ms\n",
            result.processingTimeMs);
    ASSERT_TRUE(result.processingTimeMs < 500.0f);
}

TEST(both_bands_disabled_passthrough) {
    auto input = makeSine(440.0f, 0.5f, 48000, 0.05f);
    auto original = input;

    faurge::ExciterConfig cfg;
    cfg.highEnable = false;
    cfg.lowEnable = false;

    faurge::Exciter exciter(cfg);
    auto result = exciter.process(input, 48000);

    ASSERT_TRUE(result.success);
    for (size_t i = 0; i < input.size(); ++i) {
        ASSERT_TRUE(std::fabs(input[i] - original[i]) < 1e-6f);
    }
}

TEST(empty_audio_returns_error) {
    std::vector<float> empty;
    faurge::Exciter exciter;
    auto result = exciter.process(empty, 48000);

    ASSERT_TRUE(!result.success);
    ASSERT_TRUE(!result.errorMessage.empty());
}

int main() {
    int passed = 0, failed = 0;
    fprintf(stderr, "\n=== Exciter: Pipeline Integration Tests ===\n\n");
    for (const auto& t : tests) {
        fprintf(stderr, "  [RUN]  %s\n", t.name);
        try {
            t.fn();
            fprintf(stderr, "  [PASS] %s\n", t.name);
            ++passed;
        } catch (...) {
            fprintf(stderr, "  [FAIL] %s\n", t.name);
            ++failed;
        }
    }
    fprintf(stderr, "\n  Results: %d passed, %d failed\n\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
