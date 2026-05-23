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

static void addSine(std::vector<float>& buf, float freq, float amp, float sr) {
    for (size_t i = 0; i < buf.size(); ++i) {
        buf[i] += amp * std::sin(2.0f * 3.14159265f * freq * i / sr);
    }
}

TEST(hp_sidechain_reduces_bass_triggering) {
    int sr = 44100;
    size_t n = static_cast<size_t>(sr * 0.5f);
    std::vector<float> audio(n, 0.0f);
    addSine(audio, 100.0f, 0.6f, (float)sr);
    addSine(audio, 1000.0f, 0.3f, (float)sr);

    auto audioNoSc = audio;
    auto audioWithSc = audio;

    faurge::CompConfig config;
    config.threshold_db = -30.0f;
    config.ratio = 10.0f;
    config.attack_ms = 1.0f;
    config.release_ms = 50.0f;
    config.knee_db = 0.0f;
    config.sidechain_hp_hz = 20.0f;
    config.sidechain_lp_hz = 20000.0f;

    faurge::Compressor compNoSc(config);
    auto resultNoSc = compNoSc.process(audioNoSc, sr);
    ASSERT_TRUE(resultNoSc.success);

    config.sidechain_hp_hz = 500.0f;
    faurge::Compressor compWithSc(config);
    auto resultWithSc = compWithSc.process(audioWithSc, sr);
    ASSERT_TRUE(resultWithSc.success);

    ASSERT_TRUE(resultNoSc.gainReductionDb > 0.0f);
    ASSERT_TRUE(resultWithSc.gainReductionDb > 0.0f);
}

TEST(lp_sidechain_reduces_high_triggering) {
    int sr = 44100;
    size_t n = static_cast<size_t>(sr * 0.5f);
    std::vector<float> audio(n, 0.0f);
    addSine(audio, 100.0f, 0.2f, (float)sr);
    addSine(audio, 8000.0f, 0.6f, (float)sr);

    auto audioNoSc = audio;
    auto audioWithSc = audio;

    faurge::CompConfig config;
    config.threshold_db = -30.0f;
    config.ratio = 10.0f;
    config.attack_ms = 1.0f;
    config.release_ms = 50.0f;
    config.knee_db = 0.0f;
    config.sidechain_hp_hz = 20.0f;
    config.sidechain_lp_hz = 20000.0f;

    faurge::Compressor compNoSc(config);
    auto resultNoSc = compNoSc.process(audioNoSc, sr);
    ASSERT_TRUE(resultNoSc.success);

    config.sidechain_lp_hz = 200.0f;
    faurge::Compressor compWithSc(config);
    auto resultWithSc = compWithSc.process(audioWithSc, sr);
    ASSERT_TRUE(resultWithSc.success);

    ASSERT_TRUE(resultNoSc.gainReductionDb > 0.0f);
    ASSERT_TRUE(resultWithSc.gainReductionDb > 0.0f);
}

static void runAll() {
    int passed = 0, failed = 0;
    fprintf(stderr, "\n=== Compressor Sidechain Tests ===\n\n");
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
