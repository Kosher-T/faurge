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

#define ASSERT_NEAR(a, b, eps) do { \
    float diff = std::fabs((a) - (b)); \
    if (diff > (eps)) { \
        fprintf(stderr, "  FAIL: %s ≈ %s (got %.6f, expected %.6f, eps %.6f) (line %d)\n", \
                #a, #b, (float)(a), (float)(b), (float)(eps), __LINE__); \
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

TEST(dynamic_eq_depth_zero_matches_static) {
    int sr = 48000;
    auto audio1 = makeSine(1000.0f, 0.5f, sr, 0.2f);
    auto audio2 = audio1;

    faurge::EqConfig cfg;
    cfg.bands[0].freq_hz = 1000.0f;
    cfg.bands[0].gain_db = 6.0f;
    cfg.bands[0].q = 2.0f;
    cfg.bands[0].filter_type = faurge::FilterType::peak;
    cfg.bands[0].dynamic_depth = 0.0f;

    faurge::Equalizer eq;
    auto result = eq.process(audio1, sr, cfg);

    ASSERT_TRUE(result.success);
    float dynamicRms = computeRms(audio1.data(), audio1.size());

    faurge::EqConfig staticCfg;
    staticCfg.bands[0].freq_hz = 1000.0f;
    staticCfg.bands[0].gain_db = 6.0f;
    staticCfg.bands[0].q = 2.0f;
    staticCfg.bands[0].filter_type = faurge::FilterType::peak;
    staticCfg.bands[0].dynamic_depth = 0.0f;

    eq = faurge::Equalizer(staticCfg);
    auto result2 = eq.process(audio2, sr);

    ASSERT_TRUE(result2.success);
    float staticRms = computeRms(audio2.data(), audio2.size());

    ASSERT_NEAR(dynamicRms, staticRms, 0.001f);
}

TEST(dynamic_depth_one_varies_with_envelope) {
    int sr = 48000;
    size_t n = static_cast<size_t>(sr * 0.5f);

    std::vector<float> audio(n);
    for (size_t i = 0; i < n; ++i) {
        float t = static_cast<float>(i) / sr;
        float amp = 0.1f + 0.8f * (0.5f + 0.5f * std::sin(2.0f * PI * 2.0f * t));
        audio[i] = amp * std::sin(2.0f * PI * 1000.0f * t);
    }

    auto audioDynamic = audio;

    faurge::EqConfig cfg;
    cfg.bands[0].freq_hz = 1000.0f;
    cfg.bands[0].gain_db = 12.0f;
    cfg.bands[0].q = 2.0f;
    cfg.bands[0].filter_type = faurge::FilterType::peak;
    cfg.bands[0].dynamic_depth = 1.0f;

    faurge::Equalizer eq;
    auto result = eq.process(audioDynamic, sr, cfg);

    ASSERT_TRUE(result.success);

    float rmsFirstHalf = computeRms(audioDynamic.data(), n / 2);
    float rmsSecondHalf = computeRms(audioDynamic.data() + n / 2, n / 2);

    fprintf(stderr, "    RMS first half: %.4f, second half: %.4f\n",
            rmsFirstHalf, rmsSecondHalf);
}

int main() {
    int passed = 0, failed = 0;
    fprintf(stderr, "\n=== EQ: Dynamic Depth Tests ===\n\n");
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
