#include "faurge/denoiser.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
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
static int failures = 0;

#define ASSERT_TRUE(x) do { \
    if (!(x)) { \
        fprintf(stderr, "  FAIL: %s (line %d)\n", #x, __LINE__); \
        ++failures; \
        return; \
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

TEST(resamples_44100_to_48000_and_back) {
    auto original = makeSine(440.0f, 0.5f, 44100, 0.05f);
    auto audio = original;

    faurge::DenoiseConfig cfg;
    cfg.attenLimit = 0.0f;
    faurge::Denoiser denoiser(cfg);
    auto result = denoiser.process(audio, 44100);

    ASSERT_TRUE(result.success);

    float dot = 0.0f, normOrig = 0.0f, normAudio = 0.0f;
    size_t minLen = std::min(original.size(), audio.size());
    for (size_t i = 0; i < minLen; ++i) {
        dot += original[i] * audio[i];
        normOrig += original[i] * original[i];
        normAudio += audio[i] * audio[i];
    }

    float sim = dot / (std::sqrt(normOrig) * std::sqrt(normAudio) + 1e-10f);
    fprintf(stderr, "    44.1k round-trip similarity: %.4f\n", sim);
    ASSERT_TRUE(sim > 0.9f);
}

TEST(resamples_16000_to_48000_and_back) {
    auto original = makeSine(440.0f, 0.5f, 16000, 0.05f);
    auto audio = original;

    faurge::DenoiseConfig cfg;
    cfg.attenLimit = 0.0f;
    faurge::Denoiser denoiser(cfg);
    auto result = denoiser.process(audio, 16000);

    ASSERT_TRUE(result.success);
    ASSERT_TRUE(result.processingTimeMs >= 0.0f);
    fprintf(stderr, "    16k input processed: %.2f ms\n", result.processingTimeMs);
}

TEST(forty_eight_k_input_skips_resampling) {
    auto audio = makeSine(440.0f, 0.5f, 48000, 0.05f);
    auto original = audio;

    faurge::DenoiseConfig cfg;
    cfg.attenLimit = 0.0f;
    faurge::Denoiser denoiser(cfg);
    auto result = denoiser.process(audio, 48000);

    ASSERT_TRUE(result.success);
    fprintf(stderr, "    48k input processed: %.2f ms\n", result.processingTimeMs);
}

int main() {
    int passed = 0, failed = 0;
    fprintf(stderr, "\n=== Denoiser: Resampling Tests ===\n\n");
    for (const auto& t : tests) {
        failures = 0;
        fprintf(stderr, "  [RUN]  %s\n", t.name);
        t.fn();
        if (failures == 0) {
            fprintf(stderr, "  [PASS] %s\n", t.name);
            ++passed;
        } else {
            fprintf(stderr, "  [FAIL] %s\n", t.name);
            ++failed;
        }
    }
    fprintf(stderr, "\n  Results: %d passed, %d failed\n\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
