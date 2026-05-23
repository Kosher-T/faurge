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

static float computeRmsDb(const float* buf, size_t n) {
    float sumSq = 0.0f;
    for (size_t i = 0; i < n; ++i) sumSq += buf[i] * buf[i];
    float rms = std::sqrt(sumSq / n);
    if (rms < 1e-30f) return -120.0f;
    return 20.0f * std::log10(rms);
}

TEST(single_peak_band_boosts_at_center_freq) {
    int sr = 48000;
    auto input = makeSine(1000.0f, 0.5f, sr, 0.2f);
    float inputDb = computeRmsDb(input.data(), input.size());

    faurge::EqConfig cfg;
    cfg.bands[0].freq_hz = 1000.0f;
    cfg.bands[0].gain_db = 12.0f;
    cfg.bands[0].q = 2.0f;
    cfg.bands[0].filter_type = faurge::FilterType::peak;

    faurge::Equalizer eq;
    auto result = eq.process(input, sr, cfg);

    ASSERT_TRUE(result.success);
    float outputDb = computeRmsDb(input.data(), input.size());
    ASSERT_TRUE(outputDb > inputDb + 6.0f);
}

TEST(single_peak_band_cuts_at_center_freq) {
    int sr = 48000;
    auto input = makeSine(1000.0f, 0.5f, sr, 0.2f);
    float inputDb = computeRmsDb(input.data(), input.size());

    faurge::EqConfig cfg;
    cfg.bands[0].freq_hz = 1000.0f;
    cfg.bands[0].gain_db = -12.0f;
    cfg.bands[0].q = 2.0f;
    cfg.bands[0].filter_type = faurge::FilterType::peak;

    faurge::Equalizer eq;
    auto result = eq.process(input, sr, cfg);

    ASSERT_TRUE(result.success);
    float outputDb = computeRmsDb(input.data(), input.size());
    ASSERT_TRUE(outputDb < inputDb - 6.0f);
}

TEST(peak_band_does_not_affect_distant_freq) {
    int sr = 48000;
    auto input = makeSine(100.0f, 0.5f, sr, 0.2f);
    float inputDb = computeRmsDb(input.data(), input.size());

    faurge::EqConfig cfg;
    cfg.bands[0].freq_hz = 10000.0f;
    cfg.bands[0].gain_db = 12.0f;
    cfg.bands[0].q = 2.0f;
    cfg.bands[0].filter_type = faurge::FilterType::peak;

    faurge::Equalizer eq;
    auto result = eq.process(input, sr, cfg);

    ASSERT_TRUE(result.success);
    float outputDb = computeRmsDb(input.data(), input.size());
    ASSERT_TRUE(std::fabs(outputDb - inputDb) < 1.0f);
}

TEST(multiple_bands_accumulate_gain) {
    int sr = 48000;
    auto input = makeSine(1000.0f, 0.5f, sr, 0.2f);

    faurge::EqConfig cfg;
    cfg.bands[0].freq_hz = 1000.0f;
    cfg.bands[0].gain_db = 6.0f;
    cfg.bands[0].q = 2.0f;
    cfg.bands[0].filter_type = faurge::FilterType::peak;

    cfg.bands[1].freq_hz = 1000.0f;
    cfg.bands[1].gain_db = 6.0f;
    cfg.bands[1].q = 2.0f;
    cfg.bands[1].filter_type = faurge::FilterType::peak;

    faurge::Equalizer eq;
    auto result = eq.process(input, sr, cfg);

    ASSERT_TRUE(result.success);
    float outputDb = computeRmsDb(input.data(), input.size());
    float inputDb = computeRmsDb(
        makeSine(1000.0f, 0.5f, sr, 0.2f).data(),
        static_cast<size_t>(sr * 0.2f));
    ASSERT_TRUE(outputDb > inputDb + 8.0f);
}

int main() {
    int passed = 0, failed = 0;
    fprintf(stderr, "\n=== EQ: Sine Sweep Tests ===\n\n");
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
