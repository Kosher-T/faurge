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

TEST(dynamic_gate_has_measurable_gr_on_sibilant_tone) {
    int sr = 48000;
    auto input = makeSine(7000.0f, 0.5f, sr, 1.0f);

    faurge::EsserConfig cfg;
    cfg.center_freq_hz = 7000.0f;
    cfg.threshold_db = -25.0f;
    cfg.ratio = 10.0f;
    cfg.bandwidth_hz = 2000.0f;
    cfg.attack_ms = 1.0f;
    cfg.release_ms = 50.0f;

    auto proc = input;
    faurge::Esser esser(cfg);
    auto result = esser.process(proc, sr, cfg);
    ASSERT_TRUE(result.success);
    ASSERT_TRUE(result.maxGainReductionDb > 6.0f);
    ASSERT_TRUE(result.sibilantFrames > 0);
}

TEST(dynamic_gate_reduces_output_level) {
    int sr = 48000;
    auto input = makeSine(7000.0f, 0.5f, sr, 1.0f);
    float inputRms = computeRmsDb(input.data(), input.size());

    faurge::EsserConfig cfg;
    cfg.center_freq_hz = 7000.0f;
    cfg.threshold_db = -25.0f;
    cfg.ratio = 10.0f;
    cfg.bandwidth_hz = 2000.0f;
    cfg.attack_ms = 1.0f;
    cfg.release_ms = 50.0f;

    auto proc = input;
    faurge::Esser esser(cfg);
    auto result = esser.process(proc, sr, cfg);
    ASSERT_TRUE(result.success);

    float outputRms = computeRmsDb(proc.data(), proc.size());
    ASSERT_TRUE(outputRms < inputRms - 3.0f);
}

TEST(dynamic_gate_gr_depends_on_input_level) {
    int sr = 48000;

    auto quietInput = makeSine(7000.0f, 0.01f, sr, 0.5f);
    auto loudInput = makeSine(7000.0f, 0.5f, sr, 0.5f);

    faurge::EsserConfig cfg;
    cfg.center_freq_hz = 7000.0f;
    cfg.threshold_db = -25.0f;
    cfg.ratio = 10.0f;
    cfg.bandwidth_hz = 2000.0f;
    cfg.attack_ms = 1.0f;
    cfg.release_ms = 50.0f;

    faurge::Esser esser(cfg);

    auto quietCopy = quietInput;
    faurge::EsserConfig quietCfg = cfg;
    quietCfg.attack_ms = 0.1f;
    quietCfg.release_ms = 10.0f;
    auto quietResult = esser.process(quietCopy, sr, quietCfg);

    auto loudCopy = loudInput;
    auto loudResult = esser.process(loudCopy, sr, cfg);

    ASSERT_TRUE(quietResult.success);
    ASSERT_TRUE(loudResult.success);

    ASSERT_TRUE(loudResult.maxGainReductionDb > quietResult.maxGainReductionDb + 5.0f);
}

int main() {
    int passed = 0, failed = 0;
    fprintf(stderr, "\n=== Esser: VS EQ Tests ===\n\n");
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
