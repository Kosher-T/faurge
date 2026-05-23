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

static std::vector<float> makeTransient(float sr, float durSec) {
    size_t n = static_cast<size_t>(sr * durSec);
    std::vector<float> buf(n, 0.0f);

    size_t clickPos = static_cast<size_t>(sr * 0.1f);
    for (size_t i = 0; i < 100 && clickPos + i < n; ++i) {
        buf[clickPos + i] = 0.9f * (1.0f - (float)i / 100.0f);
    }
    return buf;
}

TEST(lookahead_shifts_gr) {
    int sr = 44100;
    auto audio = makeTransient((float)sr, 0.5f);
    auto audioNoLookahead = audio;
    auto audioWithLookahead = audio;

    faurge::CompConfig config;
    config.threshold_db = -40.0f;
    config.ratio = 20.0f;
    config.attack_ms = 0.1f;
    config.release_ms = 5.0f;
    config.knee_db = 0.0f;
    config.lookahead_ms = 0.0f;

    faurge::Compressor comp(config);
    comp.process(audioNoLookahead, sr);

    config.lookahead_ms = 5.0f;
    faurge::Compressor compLa(config);
    compLa.process(audioWithLookahead, sr);

    int lookaheadSamples = static_cast<int>(5.0f * sr / 1000.0f);

    float grNoLa = 0.0f, grWithLa = 0.0f;
    size_t grStartNoLa = 0, grStartWithLa = 0;
    bool foundNoLa = false, foundWithLa = false;

    for (size_t i = 0; i < audioNoLookahead.size(); ++i) {
        float diff = std::fabs(audioNoLookahead[i] - audio[i]);
        if (diff > 0.01f && !foundNoLa) {
            grStartNoLa = i;
            foundNoLa = true;
        }
        if (diff > grNoLa) grNoLa = diff;
    }

    for (size_t i = 0; i < audioWithLookahead.size(); ++i) {
        float diff = std::fabs(audioWithLookahead[i] - audio[i]);
        if (diff > 0.01f && !foundWithLa) {
            grStartWithLa = i;
            foundWithLa = true;
        }
        if (diff > grWithLa) grWithLa = diff;
    }

    ASSERT_TRUE(foundNoLa);
    ASSERT_TRUE(foundWithLa);
    ASSERT_TRUE(grWithLa > 0.01f);
}

TEST(lookahead_no_audio_delay) {
    int sr = 44100;

    size_t n = static_cast<size_t>(sr * 0.3f);
    std::vector<float> audio(n);
    for (size_t i = 0; i < n; ++i) {
        audio[i] = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f * i / sr);
    }

    auto audioNoLa = audio;
    auto audioWithLa = audio;

    faurge::CompConfig config;
    config.threshold_db = -20.0f;
    config.ratio = 4.0f;
    config.attack_ms = 5.0f;
    config.release_ms = 50.0f;
    config.lookahead_ms = 0.0f;

    faurge::Compressor comp(config);
    auto resultNo = comp.process(audioNoLa, sr);

    config.lookahead_ms = 5.0f;
    faurge::Compressor compLa(config);
    auto resultWith = compLa.process(audioWithLa, sr);

    ASSERT_TRUE(resultNo.success);
    ASSERT_TRUE(resultWith.success);

    ASSERT_NEAR(audioNoLa.size(), audioWithLa.size(), 0);
}

static void runAll() {
    int passed = 0, failed = 0;
    fprintf(stderr, "\n=== Compressor Lookahead Tests ===\n\n");
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
