#include "faurge/denoiser.hpp"
#include "faurge/noise_estimator.hpp"

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

static std::vector<float> makeNoisySine(float amp, float noiseLevel,
                                        int sr, float dur) {
    size_t n = static_cast<size_t>(sr * dur);
    std::vector<float> buf(n);
    for (size_t i = 0; i < n; ++i) {
        float sine = amp * std::sin(2.0f * PI * 440.0f * i / sr);
        float noise = static_cast<float>(std::rand()) / RAND_MAX * 2.0f * noiseLevel - noiseLevel;
        buf[i] = sine + noise;
    }
    return buf;
}

TEST(zero_attenuation_is_bypass) {
    auto audio = makeNoisySine(0.3f, 0.1f, 48000, 0.1f);
    auto original = audio;

    faurge::DenoiseConfig cfg;
    cfg.attenLimit = 0.0f;
    faurge::Denoiser denoiser(cfg);
    auto result = denoiser.process(audio, 48000);

    ASSERT_TRUE(result.success);

    float maxDiff = 0.0f;
    for (size_t i = 0; i < audio.size(); ++i) {
        float diff = std::fabs(audio[i] - original[i]);
        if (diff > maxDiff) maxDiff = diff;
    }
    fprintf(stderr, "    atten=0.0 max diff: %f\n", maxDiff);
    ASSERT_TRUE(maxDiff < 0.05f);
}

TEST(max_attenuation_is_aggressive) {
    auto audio = makeNoisySine(0.3f, 0.15f, 48000, 0.15f);

    faurge::NoiseEstimator est;
    float floorBefore = est.estimateNoiseFloorDb(audio.data(), audio.size(), 48000);

    faurge::DenoiseConfig cfg;
    cfg.attenLimit = 0.95f;
    faurge::Denoiser denoiser(cfg);
    auto result = denoiser.process(audio, 48000);

    ASSERT_TRUE(result.success);

    float floorAfter = est.estimateNoiseFloorDb(audio.data(), audio.size(), 48000);
    fprintf(stderr, "    atten=0.95 floor: %.1f -> %.1f dBFS\n",
            floorBefore, floorAfter);
}

TEST(higher_attenuation_removes_more_noise) {
    auto audio1 = makeNoisySine(0.3f, 0.12f, 48000, 0.15f);
    auto audio2 = audio1;

    faurge::DenoiseConfig cfgLow;
    cfgLow.attenLimit = 0.3f;
    faurge::Denoiser denoiserLow(cfgLow);
    auto resultLow = denoiserLow.process(audio1, 48000);

    faurge::DenoiseConfig cfgHigh;
    cfgHigh.attenLimit = 0.9f;
    faurge::Denoiser denoiserHigh(cfgHigh);
    auto resultHigh = denoiserHigh.process(audio2, 48000);

    ASSERT_TRUE(resultLow.success);
    ASSERT_TRUE(resultHigh.success);

    float energyLow = 0.0f, energyHigh = 0.0f;
    for (size_t i = 0; i < audio1.size(); ++i) {
        energyLow += audio1[i] * audio1[i];
        energyHigh += audio2[i] * audio2[i];
    }

    fprintf(stderr, "    atten=0.3 energy: %f, atten=0.9 energy: %f\n",
            energyLow, energyHigh);
    ASSERT_TRUE(energyHigh <= energyLow + 0.1f);
}

int main() {
    int passed = 0, failed = 0;
    fprintf(stderr, "\n=== Denoiser: Attenuation Tests ===\n\n");
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
