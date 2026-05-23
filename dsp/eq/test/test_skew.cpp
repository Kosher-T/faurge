#include "faurge/eq.hpp"

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

static std::vector<float> makeSine(float freq, float amp, int sr, float dur) {
    size_t n = static_cast<size_t>(sr * dur);
    std::vector<float> buf(n);
    for (size_t i = 0; i < n; ++i)
        buf[i] = amp * std::sin(2.0f * PI * freq * i / sr);
    return buf;
}

static float computeRms(const float* buf, size_t n) {
    float sumSq = 0.0f;
    for (size_t i = 0; i < n; ++i) sumSq += buf[i] * buf[i];
    return std::sqrt(sumSq / n);
}

TEST(stereo_skew_produces_LR_difference) {
    int sr = 48000;
    size_t n = static_cast<size_t>(sr * 0.2f);
    auto left = makeSine(1000.0f, 0.5f, sr, 0.2f);
    auto right = makeSine(1000.0f, 0.5f, sr, 0.2f);

    faurge::EqConfig cfg;
    cfg.bands[0].freq_hz = 1000.0f;
    cfg.bands[0].gain_db = 0.0f;
    cfg.bands[0].q = 2.0f;
    cfg.bands[0].filter_type = faurge::FilterType::peak;
    cfg.bands[0].stereo_skew_db = 6.0f;

    faurge::Equalizer eq;
    auto resultL = eq.process(left, sr, cfg, 0);
    auto resultR = eq.process(right, sr, cfg, 1);

    ASSERT_TRUE(resultL.success);
    ASSERT_TRUE(resultR.success);

    float leftRms = computeRms(left.data(), n);
    float rightRms = computeRms(right.data(), n);

    fprintf(stderr, "    Left RMS: %.4f, Right RMS: %.4f\n", leftRms, rightRms);
    ASSERT_TRUE(leftRms > rightRms);
}

TEST(stereo_skew_difference_only_in_band_range) {
    int sr = 48000;
    size_t n = static_cast<size_t>(sr * 0.2f);

    auto leftLow = makeSine(100.0f, 0.5f, sr, 0.2f);
    auto rightLow = leftLow;

    faurge::EqConfig cfg;
    cfg.bands[0].freq_hz = 1000.0f;
    cfg.bands[0].gain_db = 0.0f;
    cfg.bands[0].q = 2.0f;
    cfg.bands[0].filter_type = faurge::FilterType::peak;
    cfg.bands[0].stereo_skew_db = 6.0f;

    faurge::Equalizer eq;
    eq.process(leftLow, sr, cfg, 0);
    eq.process(rightLow, sr, cfg, 1);

    float leftRmsLow = computeRms(leftLow.data(), n);
    float rightRmsLow = computeRms(rightLow.data(), n);

    fprintf(stderr, "    Low: Left RMS=%.4f, Right RMS=%.4f (diff should be small)\n",
            leftRmsLow, rightRmsLow);
    ASSERT_TRUE(std::fabs(leftRmsLow - rightRmsLow) < 0.05f);
}

int main() {
    int passed = 0, failed = 0;
    fprintf(stderr, "\n=== EQ: Stereo Skew Tests ===\n\n");
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
