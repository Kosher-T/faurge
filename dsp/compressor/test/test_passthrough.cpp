#include "faurge/compressor.hpp"
#include "faurge/comp_metrics.hpp"

#include <algorithm>
#include <cassert>
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

#define ASSERT_TRUE(x) do { \
    if (!(x)) { \
        fprintf(stderr, "  FAIL: %s (line %d)\n", #x, __LINE__); \
        assert(false); \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps) do { \
    float diff = std::fabs((float)(a) - (float)(b)); \
    if (diff > (float)(eps)) { \
        fprintf(stderr, "  FAIL: |%s - %s| = %g > %g (line %d)\n", \
                #a, #b, diff, (float)(eps), __LINE__); \
        assert(false); \
    } \
} while(0)

static std::vector<float> makeSine(float freq, float amp, float sr, float durSec) {
    size_t n = static_cast<size_t>(sr * durSec);
    std::vector<float> buf(n);
    for (size_t i = 0; i < n; ++i) {
        buf[i] = amp * std::sin(2.0f * 3.14159265f * freq * i / sr);
    }
    return buf;
}

TEST(ratio_one) {
    int sr = 44100;
    auto audio = makeSine(440.0f, 0.5f, (float)sr, 1.0f);
    auto original = audio;

    faurge::CompConfig config;
    config.ratio = 1.0f;
    config.threshold_db = -80.0f;
    config.attack_ms = 0.1f;
    config.release_ms = 10.0f;
    config.knee_db = 0.0f;

    faurge::Compressor comp(config);
    auto result = comp.process(audio, sr);

    ASSERT_TRUE(result.success);
    ASSERT_NEAR(result.gainReductionDb, 0.0f, 0.1f);

    for (size_t i = 0; i < audio.size(); ++i) {
        ASSERT_NEAR(audio[i], original[i], 1e-6f);
    }
}

TEST(threshold_above_peak) {
    int sr = 44100;
    auto audio = makeSine(440.0f, 0.5f, (float)sr, 1.0f);

    faurge::CompConfig config;
    config.threshold_db = 0.0f;
    config.ratio = 20.0f;
    config.attack_ms = 0.1f;
    config.release_ms = 10.0f;
    config.knee_db = 0.0f;

    faurge::Compressor comp(config);
    auto result = comp.process(audio, sr);

    ASSERT_TRUE(result.success);
    ASSERT_NEAR(result.gainReductionDb, 0.0f, 0.1f);
}

TEST(wet_dry_zero) {
    int sr = 44100;
    auto audio = makeSine(440.0f, 0.5f, (float)sr, 1.0f);
    auto original = audio;

    faurge::CompConfig config;
    config.threshold_db = -60.0f;
    config.ratio = 20.0f;
    config.attack_ms = 0.1f;
    config.release_ms = 10.0f;
    config.knee_db = 0.0f;
    config.wet_dry_mix = 0.0f;

    faurge::Compressor comp(config);
    auto result = comp.process(audio, sr);

    ASSERT_TRUE(result.success);

    for (size_t i = 0; i < audio.size(); ++i) {
        ASSERT_NEAR(audio[i], original[i], 1e-6f);
    }
}

static void runAll() {
    int passed = 0, failed = 0;
    fprintf(stderr, "\n=== Compressor Passthrough Tests ===\n\n");
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
    std::exit(failed > 0 ? 1 : 0);
}

int main() {
    runAll();
    return 0;
}
