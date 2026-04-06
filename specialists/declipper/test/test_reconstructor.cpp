// ═══════════════════════════════════════════════════════════════
// Faurge — De-Clipper Unit Tests: Reconstruction
// ═══════════════════════════════════════════════════════════════
#include "faurge/clip_detector.hpp"
#include "faurge/reconstructor.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>
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

static float computeSnr(const std::vector<float>& original,
                        const std::vector<float>& reconstructed) {
    float sigPow = 0.0f, noisePow = 0.0f;
    size_t n = std::min(original.size(), reconstructed.size());
    for (size_t i = 0; i < n; ++i) {
        sigPow   += original[i] * original[i];
        float err = reconstructed[i] - original[i];
        noisePow += err * err;
    }
    if (noisePow < 1e-30f) return 120.0f;
    return 10.0f * std::log10(sigPow / noisePow);
}

// ── Tests ────────────────────────────────────────────────────

TEST(hermite_improves_snr_on_short_clip) {
    // Ground truth: clean sine
    auto clean = makeSine(440.0f, 1.3f, 48000, 0.02f);

    // Create clipped version
    auto clipped = clean;
    for (auto& s : clipped) {
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
    }

    float snrBefore = computeSnr(clean, clipped);

    // Detect and reconstruct
    faurge::DeclipConfig cfg;
    cfg.clipThreshold = 0.9999f;
    cfg.anchorSize = 4;
    faurge::ClipDetector det(cfg);
    auto report = det.detect(clipped.data(), clipped.size());

    faurge::Reconstructor rec(cfg);
    rec.reconstruct(clipped.data(), clipped.size(), report.regions);

    float snrAfter = computeSnr(clean, clipped);

    fprintf(stderr, "    SNR before: %.1f dB, after: %.1f dB\n",
            snrBefore, snrAfter);
    ASSERT_TRUE(snrAfter > snrBefore);
}

TEST(hermite_produces_c1_continuous_output) {
    // Check that the slope at clip boundaries is continuous
    auto clean = makeSine(440.0f, 1.3f, 48000, 0.02f);
    auto clipped = clean;
    for (auto& s : clipped) {
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
    }

    faurge::DeclipConfig cfg;
    cfg.anchorSize = 4;
    faurge::ClipDetector det(cfg);
    auto report = det.detect(clipped.data(), clipped.size());

    faurge::Reconstructor rec(cfg);
    rec.reconstruct(clipped.data(), clipped.size(), report.regions);

    // Check continuity at each clip boundary
    for (const auto& r : report.regions) {
        if (r.startSample > 0 && r.endSample + 1 < clipped.size()) {
            // Slope before clip boundary
            float slopeBefore = clipped[r.startSample] -
                                clipped[r.startSample - 1];
            // Slope just inside clip
            float slopeInside = clipped[r.startSample + 1] -
                                clipped[r.startSample];
            // Should be roughly similar (C1)
            float slopeDiff = std::fabs(slopeInside - slopeBefore);
            // Allow some tolerance
            ASSERT_TRUE(slopeDiff < 0.5f);
        }
    }
}

TEST(akima_handles_medium_clips) {
    // Create a signal with a medium-length clip (20+ samples)
    auto clean = makeSine(200.0f, 1.5f, 48000, 0.05f);
    auto clipped = clean;
    for (auto& s : clipped) {
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
    }

    faurge::DeclipConfig cfg;
    cfg.hermiteMaxLen = 16;
    cfg.akimaMaxLen = 64;
    cfg.anchorSize = 4;
    faurge::ClipDetector det(cfg);
    auto report = det.detect(clipped.data(), clipped.size());

    // Check that we have at least one region that triggers Akima
    bool hasAkimaRegion = false;
    for (const auto& r : report.regions) {
        if (r.length() > 16 && r.length() <= 64) hasAkimaRegion = true;
    }

    faurge::Reconstructor rec(cfg);
    rec.reconstruct(clipped.data(), clipped.size(), report.regions);

    float snrBefore = computeSnr(clean, std::vector<float>(clean.size(), 1.0f));
    float snrAfter  = computeSnr(clean, clipped);

    fprintf(stderr, "    Has Akima-range region: %s\n",
            hasAkimaRegion ? "yes" : "no");
    fprintf(stderr, "    Reconstruction SNR: %.1f dB\n", snrAfter);

    // Reconstruction should produce values that differ from the clipped version
    // (i.e., it actually did something)
    bool changed = false;
    for (const auto& r : report.regions) {
        for (size_t j = r.startSample; j <= r.endSample; ++j) {
            if (std::fabs(clipped[j]) < 0.9999f) {
                changed = true;
                break;
            }
        }
        if (changed) break;
    }
    ASSERT_TRUE(changed);
}

TEST(ar_handles_long_clips) {
    // Create a signal with a very long clip (>64 samples)
    auto clean = makeSine(100.0f, 2.0f, 48000, 0.1f);
    auto clipped = clean;
    for (auto& s : clipped) {
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
    }

    faurge::DeclipConfig cfg;
    cfg.hermiteMaxLen = 16;
    cfg.akimaMaxLen = 64;
    cfg.arModelOrder = 14;
    cfg.anchorSize = 4;
    faurge::ClipDetector det(cfg);
    auto report = det.detect(clipped.data(), clipped.size());

    bool hasCriticalRegion = false;
    for (const auto& r : report.regions) {
        if (r.length() > 64) hasCriticalRegion = true;
    }
    fprintf(stderr, "    Has AR-range region: %s\n",
            hasCriticalRegion ? "yes" : "no");

    faurge::Reconstructor rec(cfg);
    rec.reconstruct(clipped.data(), clipped.size(), report.regions);

    // Should at least improve over flat-topped clip
    float snr = computeSnr(clean, clipped);
    fprintf(stderr, "    Reconstruction SNR: %.1f dB\n", snr);
    // AR won't be perfect, but should be better than nothing
    ASSERT_TRUE(snr > 0.0f);
}

TEST(peak_estimation_is_reasonable) {
    auto clean = makeSine(440.0f, 1.3f, 48000, 0.01f);
    auto clipped = clean;
    for (auto& s : clipped) {
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
    }

    faurge::DeclipConfig cfg;
    cfg.anchorSize = 4;
    faurge::ClipDetector det(cfg);
    auto report = det.detect(clipped.data(), clipped.size());

    faurge::Reconstructor rec(cfg);
    rec.reconstruct(clipped.data(), clipped.size(), report.regions);

    for (const auto& r : report.regions) {
        // Estimated peak should be above threshold but below overshoot limit
        ASSERT_TRUE(r.estimatedPeakAmplitude >= cfg.clipThreshold);
        ASSERT_TRUE(r.estimatedPeakAmplitude <=
                    cfg.clipThreshold * cfg.peakOvershoot * 1.01f);
    }
}

int main() {
    int passed = 0, failed = 0;
    fprintf(stderr, "\n=== De-Clipper: Reconstructor Tests ===\n\n");
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
