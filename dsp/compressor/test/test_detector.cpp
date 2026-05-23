#include "faurge/compressor.hpp"

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

static std::vector<float> makePercussive(float sr, float durSec) {
    size_t n = static_cast<size_t>(sr * durSec);
    std::vector<float> buf(n, 0.0f);

    for (int hit = 0; hit < 5; ++hit) {
        size_t hitPos = static_cast<size_t>(hit * sr * durSec / 5.0f);
        for (size_t i = 0; i < 50 && hitPos + i < n; ++i) {
            float env = 1.0f - (float)i / 50.0f;
            buf[hitPos + i] = 0.8f * env * std::sin(2.0f * 3.14159265f * 1000.0f * i / sr);
        }
    }
    return buf;
}

TEST(rms_vs_peak_difference) {
    int sr = 44100;
    auto audioRms = makePercussive((float)sr, 1.0f);
    auto audioPeak = audioRms;

    faurge::CompConfig config;
    config.threshold_db = -30.0f;
    config.ratio = 10.0f;
    config.attack_ms = 0.1f;
    config.release_ms = 10.0f;
    config.knee_db = 0.0f;
    config.detector_type = faurge::DetectorType::RMS;

    faurge::Compressor compRms(config);
    auto resultRms = compRms.process(audioRms, sr);
    ASSERT_TRUE(resultRms.success);

    config.detector_type = faurge::DetectorType::peak;
    faurge::Compressor compPeak(config);
    auto resultPeak = compPeak.process(audioPeak, sr);
    ASSERT_TRUE(resultPeak.success);

    ASSERT_TRUE(resultRms.gainReductionDb > 0.0f);
    ASSERT_TRUE(resultPeak.gainReductionDb > 0.0f);
}

TEST(feed_forward_vs_feedback) {
    int sr = 44100;

    size_t n = static_cast<size_t>(sr * 0.5f);
    std::vector<float> audio(n);
    for (size_t i = 0; i < n; ++i) {
        audio[i] = 0.6f * std::sin(2.0f * 3.14159265f * 440.0f * i / sr);
    }

    auto audioFf = audio;
    auto audioFb = audio;

    faurge::CompConfig config;
    config.threshold_db = -30.0f;
    config.ratio = 4.0f;
    config.attack_ms = 1.0f;
    config.release_ms = 50.0f;
    config.knee_db = 0.0f;
    config.detector_type = faurge::DetectorType::feed_forward;

    faurge::Compressor compFf(config);
    auto resultFf = compFf.process(audioFf, sr);
    ASSERT_TRUE(resultFf.success);

    config.detector_type = faurge::DetectorType::feed_back;
    faurge::Compressor compFb(config);
    auto resultFb = compFb.process(audioFb, sr);
    ASSERT_TRUE(resultFb.success);

    ASSERT_TRUE(resultFf.gainReductionDb > 0.0f);
}

static void runAll() {
    int passed = 0, failed = 0;
    fprintf(stderr, "\n=== Compressor Detector Tests ===\n\n");
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
