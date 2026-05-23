#include "faurge/compressor.hpp"

#include <sndfile.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

#define ASSERT_NEAR(a, b, eps) do { \
    float diff = std::fabs((float)(a) - (float)(b)); \
    if (diff > (float)(eps)) { \
        fprintf(stderr, "  FAIL: |%s - %s| = %g > %g (line %d)\n", \
                #a, #b, diff, (float)(eps), __LINE__); \
        assert(false); \
    } \
} while(0)

static void generateWavStereo(const std::vector<float>& left,
                               const std::vector<float>& right,
                               int sampleRate, const char* path) {
    size_t frames = std::min(left.size(), right.size());
    std::vector<float> interleaved(frames * 2);
    for (size_t i = 0; i < frames; ++i) {
        interleaved[i * 2]     = left[i];
        interleaved[i * 2 + 1] = right[i];
    }

    SF_INFO info;
    std::memset(&info, 0, sizeof(info));
    info.samplerate = sampleRate;
    info.channels = 2;
    info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;

    SNDFILE* f = sf_open(path, SFM_WRITE, &info);
    if (f) {
        sf_writef_float(f, interleaved.data(), frames);
        sf_close(f);
    }
}

static void readWav(const char* path, std::vector<float>& left,
                     std::vector<float>& right, int& sampleRate) {
    SF_INFO info;
    std::memset(&info, 0, sizeof(info));

    SNDFILE* f = sf_open(path, SFM_READ, &info);
    if (!f) return;

    sampleRate = info.samplerate;
    size_t totalSamples = static_cast<size_t>(info.frames) * info.channels;
    std::vector<float> interleaved(totalSamples);
    sf_readf_float(f, interleaved.data(), info.frames);
    sf_close(f);

    left.resize(info.frames);
    right.resize(info.frames);
    for (sf_count_t i = 0; i < info.frames; ++i) {
        left[i]  = interleaved[i * info.channels];
        if (info.channels > 1) right[i] = interleaved[i * info.channels + 1];
    }
}

TEST(stereo_link_produces_valid_output) {
    int sr = 44100;
    size_t n = static_cast<size_t>(sr * 0.3f);

    std::vector<float> left(n), right(n);
    for (size_t i = 0; i < n; ++i) {
        left[i]  = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f * i / sr);
        right[i] = 0.01f * std::sin(2.0f * 3.14159265f * 440.0f * i / sr);
    }

    const char* inPath  = "/tmp/test_stereo_link_in.wav";
    const char* outPath = "/tmp/test_stereo_link_out.wav";

    generateWavStereo(left, right, sr, inPath);

    faurge::CompConfig config;
    config.threshold_db = -20.0f;
    config.ratio = 10.0f;
    config.attack_ms = 0.1f;
    config.release_ms = 10.0f;
    config.knee_db = 0.0f;
    config.stereo_link = 1.0f;

    faurge::Compressor comp(config);
    auto result = comp.processFile(inPath, outPath);

    ASSERT_TRUE(result.success);
    ASSERT_TRUE(result.gainReductionDb > 0.0f);

    std::vector<float> outLeft, outRight;
    int outSr;
    readWav(outPath, outLeft, outRight, outSr);

    ASSERT_TRUE(outLeft.size() == n);
    ASSERT_TRUE(outRight.size() == n);
    ASSERT_TRUE(outSr == sr);
}

TEST(independent_channels_process_separately) {
    int sr = 44100;
    size_t n = static_cast<size_t>(sr * 0.3f);

    std::vector<float> left(n), right(n);
    for (size_t i = 0; i < n; ++i) {
        left[i]  = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f * i / sr);
        right[i] = 0.01f * std::sin(2.0f * 3.14159265f * 440.0f * i / sr);
    }

    const char* inPath  = "/tmp/test_stereo_link_indep_in.wav";
    const char* outPath = "/tmp/test_stereo_link_indep_out.wav";

    generateWavStereo(left, right, sr, inPath);

    faurge::CompConfig config;
    config.threshold_db = -20.0f;
    config.ratio = 10.0f;
    config.attack_ms = 0.1f;
    config.release_ms = 10.0f;
    config.knee_db = 0.0f;
    config.stereo_link = 0.0f;

    faurge::Compressor comp(config);
    auto result = comp.processFile(inPath, outPath);

    ASSERT_TRUE(result.success);
    ASSERT_TRUE(result.gainReductionDb > 0.0f);
}

static void runAll() {
    int passed = 0, failed = 0;
    fprintf(stderr, "\n=== Compressor Stereo Link Tests ===\n\n");
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
    std::exit(failed > 0 ? 1 : 0);
}

int main() {
    runAll();
    return 0;
}
