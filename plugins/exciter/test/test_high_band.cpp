#include "faurge/high_band.hpp"

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

static std::vector<float> makeSine(float freq, float amp,
                                   int sr, float dur) {
    size_t n = static_cast<size_t>(sr * dur);
    std::vector<float> buf(n);
    for (size_t i = 0; i < n; ++i)
        buf[i] = amp * std::sin(2.0f * PI * freq * i / sr);
    return buf;
}

static float computeRms(const float* buf, size_t n) {
    float sumSq = 0.0f;
    for (size_t i = 0; i < n; ++i) sumSq += buf[i] * buf[i];
    return std::sqrt(sumSq / n);
}

static void computeFftMagnitudes(const float* buf, size_t n,
                                 float* magOut, size_t magLen) {
    for (size_t k = 0; k < magLen; ++k) {
        float re = 0.0f, im = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            float angle = 2.0f * PI * k * i / n;
            re += buf[i] * std::cos(angle);
            im -= buf[i] * std::sin(angle);
        }
        magOut[k] = re * re + im * im;
    }
}

TEST(high_band_bypass_at_zero_drive) {
    auto input = makeSine(500.0f, 0.5f, 48000, 0.05f);
    std::vector<float> output(input.size());

    faurge::HighBand hb;
    hb.process(input.data(), output.data(), input.size(), 48000, 0.0f);

    for (size_t i = 0; i < input.size(); ++i) {
        ASSERT_TRUE(std::fabs(output[i] - input[i]) < 1e-6f);
    }
}

TEST(high_band_generates_harmonics) {
    auto input = makeSine(500.0f, 0.5f, 48000, 0.1f);
    std::vector<float> output(input.size());

    faurge::HighBand hb;
    hb.process(input.data(), output.data(), input.size(), 48000, 12.0f);

    size_t fftSize = 4096;
    if (fftSize > input.size()) fftSize = input.size();
    size_t offset = (input.size() - fftSize) / 2;

    std::vector<float> mag(fftSize / 2, 0.0f);
    computeFftMagnitudes(output.data() + offset, fftSize, mag.data(), fftSize / 2);

    size_t fundBin = static_cast<size_t>(500.0f * fftSize / 48000.0f);
    size_t harmBin = static_cast<size_t>(1000.0f * fftSize / 48000.0f);

    if (fundBin >= fftSize / 2 || harmBin >= fftSize / 2) return;

    fprintf(stderr, "    Fundamental bin %zu mag: %.6e, Harmonic bin %zu mag: %.6e\n",
            fundBin, mag[fundBin], harmBin, mag[harmBin]);
    ASSERT_TRUE(mag[harmBin] > 1e-6f);
}

TEST(high_band_does_not_clip) {
    auto input = makeSine(500.0f, 0.5f, 48000, 0.05f);
    std::vector<float> output(input.size());

    faurge::HighBand hb;
    hb.process(input.data(), output.data(), input.size(), 48000, 24.0f);

    for (size_t i = 0; i < output.size(); ++i) {
        ASSERT_TRUE(output[i] >= -1.0f && output[i] <= 1.0f);
    }
}

TEST(high_band_no_self_oscillation_on_silence) {
    std::vector<float> input(4800, 0.0f);
    std::vector<float> output(input.size());

    faurge::HighBand hb;
    hb.process(input.data(), output.data(), input.size(), 48000, 12.0f);

    float maxVal = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        float absVal = std::fabs(output[i]);
        if (absVal > maxVal) maxVal = absVal;
    }
    fprintf(stderr, "    Max output on silence: %.6e\n", maxVal);
    ASSERT_TRUE(maxVal < 1e-6f);
}

TEST(high_band_drive_increases_energy) {
    auto input = makeSine(500.0f, 0.3f, 48000, 0.05f);
    std::vector<float> outLow(input.size()), outHigh(input.size());

    faurge::HighBand hb;

    hb.process(input.data(), outLow.data(), input.size(), 48000, 3.0f);
    hb.process(input.data(), outHigh.data(), input.size(), 48000, 12.0f);

    float rmsLow = computeRms(outLow.data(), outLow.size());
    float rmsHigh = computeRms(outHigh.data(), outHigh.size());
    fprintf(stderr, "    RMS at 3dB: %.6f, RMS at 12dB: %.6f\n", rmsLow, rmsHigh);
    ASSERT_TRUE(rmsHigh > rmsLow);
}

int main() {
    int passed = 0, failed = 0;
    fprintf(stderr, "\n=== Exciter: High-Band Tests ===\n\n");
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
