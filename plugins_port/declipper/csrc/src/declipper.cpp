#include "faurge/declipper.hpp"
#include "faurge/clip_detector.hpp"
#include "faurge/reconstructor.hpp"
#include "faurge/post_filter.hpp"
#include "faurge/metrics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace faurge {

Declipper::Declipper(const DeclipConfig& config) : cfg_(config) {}

DeclipResult Declipper::process(std::vector<float>& audio, int sampleRate) {
    DeclipResult result;
    auto t0 = std::chrono::high_resolution_clock::now();

    if (audio.empty()) {
        result.success = false;
        result.errorMessage = "Empty audio buffer";
        return result;
    }

    size_t numSamples = audio.size();

    std::vector<float> beforeAudio(audio);

    ClipDetector detector(cfg_);
    result.report = detector.detect(audio.data(), numSamples);

    if (result.report.regions.empty()) {
        result.success = true;
        result.errorMessage = "No clipping detected";
        auto t1 = std::chrono::high_resolution_clock::now();
        result.processingTimeMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
        return result;
    }

    if (cfg_.verbose) {
        fprintf(stderr, "[declipper] Detected %zu clip regions (%.1f%% of audio)\n",
                result.report.regions.size(), result.report.percentClipped);
    }

    Reconstructor reconstructor(cfg_);
    reconstructor.reconstruct(audio.data(), numSamples, result.report.regions);

    PostFilter postFilter(cfg_);
    postFilter.apply(audio.data(), numSamples, result.report.regions, sampleRate);

    Metrics metrics(cfg_);
    result.regionMetrics = metrics.compute(
        beforeAudio.data(), audio.data(), numSamples, result.report);

    result.beforeThdnDb = Metrics::estimateThdnDb(
        beforeAudio.data(), numSamples, sampleRate);
    result.afterThdnDb = Metrics::estimateThdnDb(
        audio.data(), numSamples, sampleRate);

    result.success = true;

    auto t1 = std::chrono::high_resolution_clock::now();
    result.processingTimeMs =
        std::chrono::duration<float, std::milli>(t1 - t0).count();

    return result;
}

void clip_audio_inplace(std::vector<float>& audio, float gainDb, bool softClip) {
    if (audio.empty()) return;

    float gainLinear = std::pow(10.0f, gainDb / 20.0f);
    size_t n = audio.size();

    if (softClip) {
        for (size_t i = 0; i < n; ++i) {
            audio[i] = std::tanh(audio[i] * gainLinear);
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            audio[i] = audio[i] * gainLinear;
            if (audio[i] > 1.0f) audio[i] = 1.0f;
            else if (audio[i] < -1.0f) audio[i] = -1.0f;
        }
    }
}

} // namespace faurge
