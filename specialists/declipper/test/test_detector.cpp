// ═══════════════════════════════════════════════════════════════
// Faurge — De-Clipper Unit Tests: Clip Detection
// ═══════════════════════════════════════════════════════════════
#include "faurge/clip_detector.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static constexpr float PI = 3.14159265358979f;

// ── Test helpers ─────────────────────────────────────────────
#define TEST(name) \
    static void test_##name(); \
    struct Register_##name { \
        Register_##name() { tests.push_back({#name, test_##name}); } \
    } reg_##name; \
    static void test_##name()

struct TestEntry {
    const char* name;
    void (*fn)();
};
static std::vector<TestEntry> tests;

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        fprintf(stderr, "  FAIL: %s != %s (line %d)\n", #a, #b, __LINE__); \
        assert(false); \
    } \
} while(0)

#define ASSERT_TRUE(x) do { \
    if (!(x)) { \
        fprintf(stderr, "  FAIL: %s (line %d)\n", #x, __LINE__); \
        assert(false); \
    } \
} while(0)

// Generate a sine wave
static std::vector<float> makeSine(float freq, float amplitude,
                                   int sampleRate, float durationSec) {
    size_t n = static_cast<size_t>(sampleRate * durationSec);
    std::vector<float> buf(n);
    for (size_t i = 0; i < n; ++i) {
        buf[i] = amplitude * std::sin(2.0f * PI * freq * i / sampleRate);
    }
    return buf;
}

// Hard-clip a buffer at ±threshold
static void hardClip(std::vector<float>& buf, float threshold = 1.0f) {
    for (auto& s : buf) {
        if (s >  threshold) s =  threshold;
        if (s < -threshold) s = -threshold;
    }
}

// ── Tests ────────────────────────────────────────────────────

TEST(no_clipping_detected_on_clean_signal) {
    auto sine = makeSine(440.0f, 0.5f, 48000, 0.01f);  // well below threshold
    faurge::DeclipConfig cfg;
    faurge::ClipDetector det(cfg);
    auto report = det.detect(sine.data(), sine.size());
    ASSERT_EQ(report.regions.size(), 0u);
    ASSERT_EQ(report.totalClippedSamples, 0u);
}

TEST(detects_hard_clip_on_loud_sine) {
    auto sine = makeSine(440.0f, 1.5f, 48000, 0.01f);  // overshoots ±1.0
    hardClip(sine, 1.0f);

    faurge::DeclipConfig cfg;
    cfg.clipThreshold = 0.9999f;
    faurge::ClipDetector det(cfg);
    auto report = det.detect(sine.data(), sine.size());

    ASSERT_TRUE(report.regions.size() > 0);
    ASSERT_TRUE(report.totalClippedSamples > 0);
    ASSERT_TRUE(report.percentClipped > 0.0f);
}

TEST(polarity_is_correct) {
    // Create a positive-only clip
    std::vector<float> buf(100, 0.0f);
    // Rising then clipped then falling
    for (int i = 0; i < 100; ++i) {
        buf[i] = std::sin(2.0f * PI * i / 100.0f) * 1.5f;
    }
    hardClip(buf, 1.0f);

    faurge::DeclipConfig cfg;
    cfg.clipThreshold = 0.9999f;
    cfg.minClipLength = 1;
    faurge::ClipDetector det(cfg);
    auto report = det.detect(buf.data(), buf.size());

    ASSERT_TRUE(report.regions.size() >= 1);
    // First clip should be positive (sine starts going up)
    ASSERT_EQ(report.regions[0].polarity, faurge::ClipPolarity::Positive);
}

TEST(anchors_are_filled) {
    auto sine = makeSine(440.0f, 1.5f, 48000, 0.01f);
    hardClip(sine, 1.0f);

    faurge::DeclipConfig cfg;
    cfg.anchorSize = 4;
    faurge::ClipDetector det(cfg);
    auto report = det.detect(sine.data(), sine.size());

    if (!report.regions.empty()) {
        const auto& r = report.regions[0];
        // Should have anchor samples (up to anchorSize)
        ASSERT_TRUE(r.anchorsBefore.size() > 0);
        ASSERT_TRUE(r.anchorsAfter.size() > 0);
        // Anchor values should be below threshold
        for (float v : r.anchorsBefore) {
            ASSERT_TRUE(std::fabs(v) < cfg.clipThreshold);
        }
    }
}

TEST(region_merging_works) {
    // Create two clips separated by 2 samples (should merge with gap=3)
    std::vector<float> buf(50, 0.3f);
    // First clip
    buf[10] = 1.0f; buf[11] = 1.0f; buf[12] = 1.0f;
    // Gap of 2
    buf[13] = 0.5f; buf[14] = 0.5f;
    // Second clip
    buf[15] = 1.0f; buf[16] = 1.0f; buf[17] = 1.0f;

    faurge::DeclipConfig cfg;
    cfg.clipThreshold = 0.9999f;
    cfg.mergeGap = 3;
    cfg.minClipLength = 1;
    faurge::ClipDetector det(cfg);
    auto report = det.detect(buf.data(), buf.size());

    // Two clips separated by gap ≤ mergeGap → should merge into 1
    ASSERT_EQ(report.regions.size(), 1u);
}

TEST(severity_classification) {
    ASSERT_EQ(faurge::ClipDetector::classifySeverity(1),
              faurge::ClipSeverity::Mild);
    ASSERT_EQ(faurge::ClipDetector::classifySeverity(4),
              faurge::ClipSeverity::Mild);
    ASSERT_EQ(faurge::ClipDetector::classifySeverity(5),
              faurge::ClipSeverity::Moderate);
    ASSERT_EQ(faurge::ClipDetector::classifySeverity(16),
              faurge::ClipSeverity::Moderate);
    ASSERT_EQ(faurge::ClipDetector::classifySeverity(17),
              faurge::ClipSeverity::Severe);
    ASSERT_EQ(faurge::ClipDetector::classifySeverity(64),
              faurge::ClipSeverity::Severe);
    ASSERT_EQ(faurge::ClipDetector::classifySeverity(65),
              faurge::ClipSeverity::Critical);
}

// ── Runner ───────────────────────────────────────────────────
int main() {
    int passed = 0, failed = 0;
    fprintf(stderr, "\n=== De-Clipper: Detector Tests ===\n\n");
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
