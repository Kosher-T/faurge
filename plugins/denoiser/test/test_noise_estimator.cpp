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

#define ASSERT_NEAR(a, b, eps) do { \
    float diff = std::fabs((a) - (b)); \
    if (diff > (eps)) { \
        fprintf(stderr, "  FAIL: |%s - %s| = %f > %f (line %d)\n", \
                #a, #b, diff, (eps), __LINE__); \
        ++failures; \
        return; \
    } \
} while(0)

TEST(clean_signal_has_high_snr) {
    size_t n = 48000;
    std::vector<float> buf(n, 0.0f);
    for (size_t i = 0; i < n / 2; ++i)
        buf[i] = 0.5f * std::sin(2.0f * PI * 440.0f * i / 48000.0f);

    faurge::NoiseEstimator est;
    float snr = est.estimateSnrDb(buf.data(), n, 48000);
    fprintf(stderr, "    Clean (50%% silence) SNR: %.1f dB\n", snr);
    ASSERT_TRUE(snr > 10.0f);
}

TEST(noisy_signal_has_low_snr) {
    size_t n = 48000;
    std::vector<float> buf(n);
    for (size_t i = 0; i < n; ++i) {
        float sine = 0.3f * std::sin(2.0f * PI * 440.0f * i / 48000.0f);
        float noise = static_cast<float>(std::rand()) / RAND_MAX * 0.8f - 0.4f;
        buf[i] = sine + noise;
    }

    faurge::NoiseEstimator est;
    float snr = est.estimateSnrDb(buf.data(), n, 48000);
    fprintf(stderr, "    Noisy SNR: %.1f dB\n", snr);
    ASSERT_TRUE(snr > 0.0f);
    ASSERT_TRUE(snr < 15.0f);
}

TEST(noise_floor_estimation_is_stable) {
    size_t n = 48000;
    std::vector<float> buf(n);
    for (size_t i = 0; i < n; ++i)
        buf[i] = static_cast<float>(std::rand()) / RAND_MAX * 0.1f - 0.05f;

    faurge::NoiseEstimator est;
    float floor1 = est.estimateNoiseFloorDb(buf.data(), n, 48000);
    float floor2 = est.estimateNoiseFloorDb(buf.data(), n, 48000);

    fprintf(stderr, "    Floor pass 1: %.1f dBFS\n", floor1);
    fprintf(stderr, "    Floor pass 2: %.1f dBFS\n", floor2);
    ASSERT_NEAR(floor1, floor2, 6.0f);
}

TEST(silence_has_very_low_floor) {
    size_t n = 48000;
    std::vector<float> buf(n, 0.0f);
    buf[100] = 1e-6f;

    faurge::NoiseEstimator est;
    float floor = est.estimateNoiseFloorDb(buf.data(), n, 48000);
    fprintf(stderr, "    Silence floor: %.1f dBFS\n", floor);
    ASSERT_TRUE(floor < -60.0f);
}

int main() {
    int passed = 0, failed = 0;
    fprintf(stderr, "\n=== Denoiser: Noise Estimator Tests ===\n\n");
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
