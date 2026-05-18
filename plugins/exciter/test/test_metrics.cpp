#include "faurge/exciter.hpp"
#include "faurge/exciter_metrics.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

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

TEST(json_output_is_valid) {
    faurge::ExciterResult result;
    result.success = true;
    result.processingTimeMs = 12.5f;
    result.inputPeakDb = -6.0f;
    result.outputPeakDb = -3.0f;
    result.inputRmsDb = -18.0f;
    result.outputRmsDb = -15.0f;
    result.highBandEnergyDb = -25.0f;
    result.lowBandEnergyDb = -30.0f;
    result.framesProcessed = 48000;

    std::string json = faurge::ExciterMetrics::toJson(result);

    ASSERT_TRUE(json.find("{") != std::string::npos);
    ASSERT_TRUE(json.find("}") != std::string::npos);
    ASSERT_TRUE(json.find("\"success\"") != std::string::npos);
    ASSERT_TRUE(json.find("\"processing_time_ms\"") != std::string::npos);
    ASSERT_TRUE(json.find("\"input_peak_db\"") != std::string::npos);
    ASSERT_TRUE(json.find("\"output_peak_db\"") != std::string::npos);
    ASSERT_TRUE(json.find("\"high_band_energy_db\"") != std::string::npos);
    ASSERT_TRUE(json.find("\"low_band_energy_db\"") != std::string::npos);
    ASSERT_TRUE(json.find("\"frames_processed\"") != std::string::npos);

    fprintf(stderr, "    JSON length: %zu bytes\n", json.size());
}

TEST(empty_result_json_is_valid) {
    faurge::ExciterResult result;
    std::string json = faurge::ExciterMetrics::toJson(result);

    ASSERT_TRUE(json.find("{") != std::string::npos);
    ASSERT_TRUE(json.find("}") != std::string::npos);
    ASSERT_TRUE(json.find("\"success\": false") != std::string::npos);

    fprintf(stderr, "    Empty result JSON: %s\n", json.c_str());
}

TEST(metrics_are_reasonable) {
    static constexpr float PI = 3.14159265358979f;
    size_t n = 48000;
    std::vector<float> audio(n);
    for (size_t i = 0; i < n; ++i)
        audio[i] = 0.5f * std::sin(2.0f * PI * 440.0f * i / 48000.0f);

    faurge::ExciterConfig cfg;
    cfg.highDriveDb = 6.0f;
    cfg.highMix = 0.5f;
    cfg.lowDriveDb = 3.0f;
    cfg.lowMix = 0.3f;
    cfg.lowSubLevel = 0.5f;

    faurge::Exciter exciter(cfg);
    auto result = exciter.process(audio, 48000);

    ASSERT_TRUE(result.success);
    ASSERT_TRUE(result.processingTimeMs >= 0.0f);
    ASSERT_TRUE(result.inputPeakDb <= 0.0f);
    ASSERT_TRUE(result.outputPeakDb <= 0.0f);
    ASSERT_TRUE(result.inputRmsDb <= 0.0f);
    ASSERT_TRUE(result.outputRmsDb <= 0.0f);
    ASSERT_TRUE(result.framesProcessed == n);
    ASSERT_TRUE(result.processingTimeMs < 10000.0f);
}

int main() {
    int passed = 0, failed = 0;
    fprintf(stderr, "\n=== Exciter: Metrics Tests ===\n\n");
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
