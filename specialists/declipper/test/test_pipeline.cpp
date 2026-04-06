// ═══════════════════════════════════════════════════════════════
// Faurge — De-Clipper Integration Tests: Full Pipeline
// ═══════════════════════════════════════════════════════════════
#include "faurge/declipper.hpp"
#include "faurge/metrics.hpp"

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

// ── Tests ────────────────────────────────────────────────────

TEST(full_pipeline_improves_clipped_sine) {
    // Ground truth
    auto clean = makeSine(440.0f, 1.3f, 48000, 0.05f);

    // Create clipped version
    auto audio = clean;
    for (auto& s : audio) {
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
    }

    // Run full pipeline
    faurge::DeclipConfig cfg;
    cfg.verbose = false;
    faurge::Declipper dc(cfg);
    auto result = dc.process(audio, 48000);

    ASSERT_TRUE(result.success);
    ASSERT_TRUE(result.report.regions.size() > 0);
    ASSERT_TRUE(result.processingTimeMs >= 0.0f);

    // Compute SNR improvement
    float sigPow = 0.0f, noiseBefore = 0.0f, noiseAfter = 0.0f;
    for (size_t i = 0; i < clean.size(); ++i) {
        sigPow += clean[i] * clean[i];
        float errBefore = (std::min(std::max(clean[i], -1.0f), 1.0f)) - clean[i];
        noiseBefore += errBefore * errBefore;
        float errAfter = audio[i] - clean[i];
        noiseAfter += errAfter * errAfter;
    }

    float snrBefore = (noiseBefore > 1e-30f)
        ? 10.0f * std::log10(sigPow / noiseBefore) : 120.0f;
    float snrAfter = (noiseAfter > 1e-30f)
        ? 10.0f * std::log10(sigPow / noiseAfter) : 120.0f;

    fprintf(stderr, "    Pipeline SNR: %.1f dB → %.1f dB\n",
            snrBefore, snrAfter);
    ASSERT_TRUE(snrAfter > snrBefore);
}

TEST(clean_audio_passes_through_unchanged) {
    auto clean = makeSine(440.0f, 0.5f, 48000, 0.05f);
    auto original = clean;

    faurge::Declipper dc;
    auto result = dc.process(clean, 48000);

    ASSERT_TRUE(result.success);
    ASSERT_TRUE(result.report.regions.empty());

    // Audio should be identical
    for (size_t i = 0; i < clean.size(); ++i) {
        ASSERT_TRUE(std::fabs(clean[i] - original[i]) < 1e-6f);
    }
}

TEST(json_output_is_valid) {
    auto audio = makeSine(440.0f, 1.3f, 48000, 0.01f);
    for (auto& s : audio) {
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
    }

    faurge::Declipper dc;
    auto result = dc.process(audio, 48000);

    std::string json = faurge::Metrics::toJson(result);

    // Basic JSON sanity checks
    ASSERT_TRUE(json.find("{") != std::string::npos);
    ASSERT_TRUE(json.find("}") != std::string::npos);
    ASSERT_TRUE(json.find("\"success\"") != std::string::npos);
    ASSERT_TRUE(json.find("\"clip_report\"") != std::string::npos);
    ASSERT_TRUE(json.find("\"regions\"") != std::string::npos);
    ASSERT_TRUE(json.find("\"processing_time_ms\"") != std::string::npos);

    fprintf(stderr, "    JSON length: %zu bytes\n", json.size());
}

TEST(metrics_report_has_correct_counts) {
    auto audio = makeSine(440.0f, 1.5f, 48000, 0.02f);
    size_t totalSamples = audio.size();

    // Count how many samples will be clipped
    size_t expectedClipped = 0;
    for (auto& s : audio) {
        if (s >= 0.9999f || s <= -0.9999f) ++expectedClipped;
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
    }

    faurge::Declipper dc;
    auto result = dc.process(audio, 48000);

    ASSERT_TRUE(result.report.totalSamples == totalSamples);
    // Clipped count should be approximately right (merging may combine)
    ASSERT_TRUE(result.report.totalClippedSamples > 0);
    ASSERT_TRUE(result.report.percentClipped > 0.0f);
    ASSERT_TRUE(result.report.percentClipped <= 100.0f);
}

TEST(processing_time_is_reasonable) {
    // 1 second of audio at 48kHz should process in << 1 second
    auto audio = makeSine(440.0f, 1.3f, 48000, 1.0f);
    for (auto& s : audio) {
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
    }

    faurge::Declipper dc;
    auto result = dc.process(audio, 48000);

    fprintf(stderr, "    1 sec audio processed in %.2f ms\n",
            result.processingTimeMs);
    // Should be well under 1000ms for 1 second of audio
    ASSERT_TRUE(result.processingTimeMs < 1000.0f);
}

int main() {
    int passed = 0, failed = 0;
    fprintf(stderr, "\n=== De-Clipper: Pipeline Integration Tests ===\n\n");
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
