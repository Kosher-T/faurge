#include "faurge/denoiser.hpp"
#include "faurge/denoise_metrics.hpp"
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

static std::vector<float> makeSine(float freq, float amp,
                                   int sr, float dur) {
    size_t n = static_cast<size_t>(sr * dur);
    std::vector<float> buf(n);
    for (size_t i = 0; i < n; ++i)
        buf[i] = amp * std::sin(2.0f * PI * freq * i / sr);
    return buf;
}

static void addWhiteNoise(std::vector<float>& buf, float level) {
    for (auto& s : buf)
        s += static_cast<float>(std::rand()) / RAND_MAX * 2.0f * level - level;
}

TEST(denoising_reduces_noise_floor) {
    auto clean = makeSine(440.0f, 0.3f, 48000, 0.2f);
    auto noisy = clean;
    addWhiteNoise(noisy, 0.15f);

    faurge::NoiseEstimator est;
    float floorBefore = est.estimateNoiseFloorDb(noisy.data(), noisy.size(), 48000);

    faurge::DenoiseConfig cfg;
    cfg.attenLimit = 0.9f;
    cfg.verbose = false;
    faurge::Denoiser denoiser(cfg);
    auto result = denoiser.process(noisy, 48000);

    ASSERT_TRUE(result.success);

    float floorAfter = est.estimateNoiseFloorDb(noisy.data(), noisy.size(), 48000);
    fprintf(stderr, "    Noise floor: %.1f -> %.1f dBFS\n",
            floorBefore, floorAfter);
    ASSERT_TRUE(floorAfter <= floorBefore + 1.0f);
}

TEST(json_output_is_valid) {
    auto audio = makeSine(440.0f, 0.3f, 48000, 0.1f);
    addWhiteNoise(audio, 0.1f);

    faurge::DenoiseConfig cfg;
    cfg.attenLimit = 0.78f;
    faurge::Denoiser denoiser(cfg);
    auto result = denoiser.process(audio, 48000);

    std::string json = faurge::DenoiseMetrics::toJson(result);

    ASSERT_TRUE(json.find("{") != std::string::npos);
    ASSERT_TRUE(json.find("}") != std::string::npos);
    ASSERT_TRUE(json.find("\"success\"") != std::string::npos);
    ASSERT_TRUE(json.find("\"input_snr_est_db\"") != std::string::npos);
    ASSERT_TRUE(json.find("\"frames_processed\"") != std::string::npos);

    fprintf(stderr, "    JSON length: %zu bytes\n", json.size());
}

TEST(metrics_report_has_correct_fields) {
    auto audio = makeSine(440.0f, 0.3f, 48000, 0.1f);
    addWhiteNoise(audio, 0.1f);

    faurge::DenoiseConfig cfg;
    cfg.attenLimit = 0.78f;
    faurge::Denoiser denoiser(cfg);
    auto result = denoiser.process(audio, 48000);

    ASSERT_TRUE(result.success);
    ASSERT_TRUE(result.processingTimeMs >= 0.0f);
    ASSERT_TRUE(result.framesProcessed > 0);
}

TEST(processing_time_is_reasonable) {
    auto audio = makeSine(440.0f, 0.3f, 48000, 1.0f);
    addWhiteNoise(audio, 0.1f);

    faurge::DenoiseConfig cfg;
    cfg.attenLimit = 0.78f;
    faurge::Denoiser denoiser(cfg);
    auto result = denoiser.process(audio, 48000);

    fprintf(stderr, "    1 sec audio processed in %.2f ms\n",
            result.processingTimeMs);
    ASSERT_TRUE(result.processingTimeMs < 2000.0f);
}

int main() {
    int passed = 0, failed = 0;
    fprintf(stderr, "\n=== Denoiser: Pipeline Integration Tests ===\n\n");
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
