#include "faurge/esser.hpp"

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

static std::vector<float> makeNoise(float amp, int sr, float dur) {
    size_t n = static_cast<size_t>(sr * dur);
    std::vector<float> buf(n);
    for (size_t i = 0; i < n; ++i)
        buf[i] = amp * (2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f);
    return buf;
}

static float computeRmsDb(const float* buf, size_t n) {
    float sumSq = 0.0f;
    for (size_t i = 0; i < n; ++i) sumSq += buf[i] * buf[i];
    float rms = std::sqrt(sumSq / n);
    if (rms < 1e-30f) return -120.0f;
    return 20.0f * std::log10(rms);
}

TEST(gated_tone_triggers_attenuation) {
    int sr = 48000;
    size_t halfDur = static_cast<size_t>(sr * 0.5f);

    std::vector<float> input;
    for (int seg = 0; seg < 3; ++seg) {
        auto noise = makeNoise(0.1f, sr, 0.5f);
        input.insert(input.end(), noise.begin(), noise.end());
        auto tone = makeSine(7000.0f, 0.3f, sr, 0.5f);
        input.insert(input.end(), tone.begin(), tone.end());
    }

    faurge::EsserConfig cfg;
    cfg.center_freq_hz = 7000.0f;
    cfg.threshold_db = -30.0f;
    cfg.ratio = 10.0f;
    cfg.bandwidth_hz = 2000.0f;
    cfg.attack_ms = 1.0f;
    cfg.release_ms = 50.0f;

    auto inputCopy = input;
    faurge::Esser esser(cfg);
    auto result = esser.process(inputCopy, sr, cfg);

    ASSERT_TRUE(result.success);
    ASSERT_TRUE(result.maxGainReductionDb > 4.0f);
}

TEST(no_false_triggering_on_bass) {
    int sr = 48000;
    auto input = makeSine(200.0f, 0.3f, sr, 1.0f);
    float inputPeak = computeRmsDb(input.data(), input.size());

    faurge::EsserConfig cfg;
    cfg.center_freq_hz = 7000.0f;
    cfg.threshold_db = -40.0f;
    cfg.ratio = 10.0f;
    cfg.bandwidth_hz = 2000.0f;
    cfg.attack_ms = 1.0f;
    cfg.release_ms = 50.0f;

    auto inputCopy = input;
    faurge::Esser esser(cfg);
    auto result = esser.process(inputCopy, sr, cfg);

    ASSERT_TRUE(result.success);
    float outputRms = computeRmsDb(inputCopy.data(), inputCopy.size());
    ASSERT_TRUE(std::fabs(outputRms - inputPeak) < 1.0f);
    ASSERT_TRUE(result.maxGainReductionDb < 1.0f);
}

TEST(gate_releases_after_sibilance) {
    int sr = 48000;
    std::vector<float> input;
    auto preSilence = std::vector<float>(static_cast<size_t>(sr * 0.5f), 0.0f);
    auto tone = makeSine(7000.0f, 0.3f, sr, 0.5f);
    auto postSilence = std::vector<float>(static_cast<size_t>(sr * 1.0f), 0.0f);

    input.insert(input.end(), preSilence.begin(), preSilence.end());
    input.insert(input.end(), tone.begin(), tone.end());
    input.insert(input.end(), postSilence.begin(), postSilence.end());

    size_t toneStart = preSilence.size();
    size_t toneEnd = toneStart + tone.size();

    faurge::EsserConfig cfg;
    cfg.center_freq_hz = 7000.0f;
    cfg.threshold_db = -30.0f;
    cfg.ratio = 10.0f;
    cfg.bandwidth_hz = 2000.0f;
    cfg.attack_ms = 1.0f;
    cfg.release_ms = 50.0f;

    auto inputCopy = input;
    faurge::Esser esser(cfg);
    auto result = esser.process(inputCopy, sr, cfg);
    ASSERT_TRUE(result.success);
    ASSERT_TRUE(result.maxGainReductionDb > 4.0f);
    ASSERT_TRUE(result.sibilantFrames > 0);

    float postRms = computeRmsDb(inputCopy.data() + toneEnd,
                                 inputCopy.size() - toneEnd);
    ASSERT_TRUE(postRms < -100.0f);
}

int main() {
    int passed = 0, failed = 0;
    fprintf(stderr, "\n=== Esser: Gated Tests ===\n\n");
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
