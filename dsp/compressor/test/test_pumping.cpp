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

static std::vector<float> makeToneBurst(float freq, float amp, float sr,
                                         float burstDur, float silenceDur, int cycles) {
    size_t burstSamples = static_cast<size_t>(sr * burstDur);
    size_t silenceSamples = static_cast<size_t>(sr * silenceDur);
    size_t total = (burstSamples + silenceSamples) * cycles;
    std::vector<float> buf(total, 0.0f);

    for (int c = 0; c < cycles; ++c) {
        size_t offset = c * (burstSamples + silenceSamples);
        for (size_t i = 0; i < burstSamples; ++i) {
            buf[offset + i] = amp * std::sin(2.0f * 3.14159265f * freq * i / sr);
        }
    }
    return buf;
}

TEST(extreme_pumping) {
    int sr = 44100;
    auto audio = makeToneBurst(440.0f, 0.3f, (float)sr, 0.5f, 0.5f, 3);

    faurge::CompConfig config;
    config.threshold_db = -60.0f;
    config.ratio = 20.0f;
    config.attack_ms = 0.1f;
    config.release_ms = 10.0f;
    config.knee_db = 0.0f;

    faurge::Compressor comp(config);
    auto result = comp.process(audio, sr);

    ASSERT_TRUE(result.success);
    ASSERT_TRUE(result.gainReductionDb > 10.0f);
}

TEST(hold_suppresses_pumping) {
    int sr = 44100;
    auto audio = makeToneBurst(440.0f, 0.3f, (float)sr, 0.5f, 0.2f, 3);

    faurge::CompConfig config;
    config.threshold_db = -60.0f;
    config.ratio = 20.0f;
    config.attack_ms = 0.1f;
    config.release_ms = 200.0f;
    config.knee_db = 0.0f;
    config.hold_ms = 100.0f;

    faurge::Compressor comp(config);
    auto result = comp.process(audio, sr);

    ASSERT_TRUE(result.success);
    ASSERT_TRUE(result.gainReductionDb > 10.0f);
}

static void runAll() {
    int passed = 0, failed = 0;
    fprintf(stderr, "\n=== Compressor Pumping Tests ===\n\n");
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
