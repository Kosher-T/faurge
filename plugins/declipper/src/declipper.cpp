// ═══════════════════════════════════════════════════════════════
// Faurge Specialist — De-Clipper: Main Orchestrator
// ═══════════════════════════════════════════════════════════════
// Wires all stages together:
//   Detection → Reconstruction → Post-Filter → Metrics
//
// Handles both in-memory processing and WAV file I/O via
// libsndfile.
// ═══════════════════════════════════════════════════════════════
#include "faurge/declipper.hpp"
#include "faurge/clip_detector.hpp"
#include "faurge/reconstructor.hpp"
#include "faurge/post_filter.hpp"
#include "faurge/metrics.hpp"

#include <sndfile.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace faurge {

Declipper::Declipper(const DeclipConfig& config) : cfg_(config) {}

// ── In-memory processing ─────────────────────────────────────
DeclipResult Declipper::process(std::vector<float>& audio, int sampleRate) {
    DeclipResult result;
    auto t0 = std::chrono::high_resolution_clock::now();

    if (audio.empty()) {
        result.success = false;
        result.errorMessage = "Empty audio buffer";
        return result;
    }

    size_t numSamples = audio.size();

    // Keep a copy of the original for metrics comparison
    std::vector<float> beforeAudio(audio);

    // ── Stage 1: Detection ───────────────────────────────────
    ClipDetector detector(cfg_);
    result.report = detector.detect(audio.data(), numSamples);

    if (result.report.regions.empty()) {
        // No clipping detected — nothing to do
        result.success = true;
        result.errorMessage = "No clipping detected";
        auto t1 = std::chrono::high_resolution_clock::now();
        result.processingTimeMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
        return result;
    }

    if (cfg_.verbose) {
        fprintf(stderr, "[declipper] Detected %zu clip regions "
                        "(%.1f%% of audio)\n",
                result.report.regions.size(),
                result.report.percentClipped);
    }

    // ── Stage 2+3: Reconstruction ────────────────────────────
    Reconstructor reconstructor(cfg_);
    reconstructor.reconstruct(audio.data(), numSamples,
                              result.report.regions);

    // ── Stage 4: Post-processing ─────────────────────────────
    PostFilter postFilter(cfg_);
    postFilter.apply(audio.data(), numSamples,
                     result.report.regions, sampleRate);

    // ── Stage 5: Metrics ─────────────────────────────────────
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

// ── File-based processing ────────────────────────────────────
DeclipResult Declipper::processFile(const std::string& inputPath,
                                    const std::string& outputPath) {
    DeclipResult result;

    // ── Read input WAV ───────────────────────────────────────
    SF_INFO sfInfoIn;
    std::memset(&sfInfoIn, 0, sizeof(sfInfoIn));

    SNDFILE* inFile = sf_open(inputPath.c_str(), SFM_READ, &sfInfoIn);
    if (!inFile) {
        result.success = false;
        result.errorMessage = std::string("Failed to open input: ")
                            + sf_strerror(nullptr);
        return result;
    }

    int channels   = sfInfoIn.channels;
    int sampleRate = sfInfoIn.samplerate;
    sf_count_t totalFrames = sfInfoIn.frames;

    if (cfg_.verbose) {
        fprintf(stderr, "[declipper] Input: %s\n", inputPath.c_str());
        fprintf(stderr, "[declipper]   Channels:    %d\n", channels);
        fprintf(stderr, "[declipper]   Sample rate: %d Hz\n", sampleRate);
        fprintf(stderr, "[declipper]   Frames:      %lld\n",
                static_cast<long long>(totalFrames));
    }

    // Read interleaved samples
    size_t totalSamples = static_cast<size_t>(totalFrames) * channels;
    std::vector<float> interleavedBuf(totalSamples);
    sf_count_t framesRead = sf_readf_float(inFile, interleavedBuf.data(),
                                           totalFrames);
    sf_close(inFile);

    if (framesRead != totalFrames) {
        result.success = false;
        result.errorMessage = "Failed to read all frames from input";
        return result;
    }

    // ── Process each channel independently ───────────────────
    // De-interleave → process → re-interleave
    std::vector<std::vector<float>> channelBufs(channels);
    for (int ch = 0; ch < channels; ++ch) {
        channelBufs[ch].resize(totalFrames);
        for (sf_count_t f = 0; f < totalFrames; ++f) {
            channelBufs[ch][f] = interleavedBuf[f * channels + ch];
        }
    }

    // Process each channel.  Aggregate results from channel 0 as primary.
    DeclipResult aggregateResult;
    for (int ch = 0; ch < channels; ++ch) {
        if (cfg_.verbose) {
            fprintf(stderr, "[declipper] Processing channel %d/%d...\n",
                    ch + 1, channels);
        }

        auto chResult = process(channelBufs[ch], sampleRate);

        if (ch == 0) {
            aggregateResult = chResult;
        } else {
            // Merge metrics
            aggregateResult.report.totalClippedSamples +=
                chResult.report.totalClippedSamples;
            for (auto& rm : chResult.regionMetrics) {
                aggregateResult.regionMetrics.push_back(rm);
            }
            aggregateResult.report.regions.insert(
                aggregateResult.report.regions.end(),
                chResult.report.regions.begin(),
                chResult.report.regions.end());
        }
    }

    // Re-interleave
    for (int ch = 0; ch < channels; ++ch) {
        for (sf_count_t f = 0; f < totalFrames; ++f) {
            interleavedBuf[f * channels + ch] = channelBufs[ch][f];
        }
    }

    // ── Write output WAV ─────────────────────────────────────
    SF_INFO sfInfoOut = sfInfoIn;
    sfInfoOut.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;

    SNDFILE* outFile = sf_open(outputPath.c_str(), SFM_WRITE, &sfInfoOut);
    if (!outFile) {
        aggregateResult.success = false;
        aggregateResult.errorMessage = std::string("Failed to open output: ")
                                     + sf_strerror(nullptr);
        return aggregateResult;
    }

    sf_writef_float(outFile, interleavedBuf.data(), totalFrames);
    sf_close(outFile);

    if (cfg_.verbose) {
        fprintf(stderr, "[declipper] Output written: %s\n",
                outputPath.c_str());
    }

    return aggregateResult;
}

}  // namespace faurge
