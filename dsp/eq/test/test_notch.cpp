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
        fprintf(stderr, "  FAIL: %s ≈ %s (got %.4f, expected %.4f, eps %.4f) (line %d)\n", \
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

TEST(notch_creates_null_at_1kHz) {
    int sr = 48000;
    auto atNotch = makeSine(1000.0f, 0.5f, sr, 0.3f);
    float originalRms = computeRms(atNotch.data(), atNotch.size());

    faurge::EqConfig cfg;
    cfg.bands[0].freq_hz = 1000.0f;
    cfg.bands[0].gain_db = 0.0f;
    cfg.bands[0].q = 10.0f;
    cfg.bands[0].filter_type = faurge::FilterType::notch;

    faurge::Equalizer eq;
    auto result = eq.process(atNotch, sr, cfg);

    ASSERT_TRUE(result.success);
    float notchedRms = computeRms(atNotch.data(), atNotch.size());
    ASSERT_TRUE(notchedRms < originalRms * 0.3f);
}

TEST(notch_leaves_distant_freq_untouched) {
    int sr = 48000;
    auto farFromNotch = makeSine(100.0f, 0.5f, sr, 0.3f);
    float originalRms = computeRms(farFromNotch.data(), farFromNotch.size());

    faurge::EqConfig cfg;
    cfg.bands[0].freq_hz = 1000.0f;
    cfg.bands[0].gain_db = 0.0f;
    cfg.bands[0].q = 10.0f;
    cfg.bands[0].filter_type = faurge::FilterType::notch;

    faurge::Equalizer eq;
    auto result = eq.process(farFromNotch, sr, cfg);

    ASSERT_TRUE(result.success);
    float outputRms = computeRms(farFromNotch.data(), farFromNotch.size());
    ASSERT_NEAR(outputRms, originalRms, 0.01f);
}

TEST(notch_multiple_freqs_all_notched) {
    int sr = 48000;
    size_t n = static_cast<size_t>(sr * 0.3f);
    std::vector<float> multiTone(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        float t = static_cast<float>(i) / sr;
        multiTone[i] = 0.3f * (std::sin(2.0f * PI * 500.0f * t)
                              + std::sin(2.0f * PI * 1000.0f * t)
                              + std::sin(2.0f * PI * 2000.0f * t));
    }
    float originalRms = computeRms(multiTone.data(), n);

    faurge::EqConfig cfg;
    cfg.bands[0].freq_hz = 500.0f;
    cfg.bands[0].gain_db = 0.0f;
    cfg.bands[0].q = 20.0f;
    cfg.bands[0].filter_type = faurge::FilterType::notch;
    cfg.bands[1].freq_hz = 1000.0f;
    cfg.bands[1].gain_db = 0.0f;
    cfg.bands[1].q = 20.0f;
    cfg.bands[1].filter_type = faurge::FilterType::notch;
    cfg.bands[2].freq_hz = 2000.0f;
    cfg.bands[2].gain_db = 0.0f;
    cfg.bands[2].q = 20.0f;
    cfg.bands[2].filter_type = faurge::FilterType::notch;

    faurge::Equalizer eq;
    auto result = eq.process(multiTone, sr, cfg);

    ASSERT_TRUE(result.success);
    float notchedRms = computeRms(multiTone.data(), n);
    ASSERT_TRUE(notchedRms < originalRms * 0.5f);
}

int main() {
    int passed = 0, failed = 0;
    fprintf(stderr, "\n=== EQ: Notch Tests ===\n\n");
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
