#include "faurge/denoiser.hpp"
#include "faurge/noise_estimator.hpp"
#include "faurge/denoise_metrics.hpp"

#include <sndfile.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" {
    void* df_bridge_create(const char* model_path, float atten_limit);
    int   df_bridge_process(void* handle, float* samples, size_t num_samples);
    size_t df_bridge_get_frame_size(void* handle);
    size_t df_bridge_get_sample_rate(void* handle);
    void  df_bridge_set_atten_limit(void* handle, float limit);
    void  df_bridge_destroy(void* handle);
}

namespace faurge {

struct Denoiser::Impl {
    void* bridgeHandle = nullptr;
    DenoiseConfig cfg;
    NoiseEstimator estimator;
    int nativeSampleRate = 48000;
    size_t frameSize = 0;

    ~Impl() {
        if (bridgeHandle) {
            df_bridge_destroy(bridgeHandle);
            bridgeHandle = nullptr;
        }
    }
};

Denoiser::Denoiser(const DenoiseConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->cfg = config;
    const char* modelPath = config.modelPath.empty() ? nullptr : config.modelPath.c_str();
    impl_->bridgeHandle = df_bridge_create(modelPath, config.attenLimit);
    if (impl_->bridgeHandle) {
        impl_->frameSize = df_bridge_get_frame_size(impl_->bridgeHandle);
    }
}

Denoiser::~Denoiser() = default;

const DenoiseConfig& Denoiser::config() const {
    return impl_->cfg;
}

static void resampleBuffer(const std::vector<float>& input,
                           int inputSampleRate,
                           int outputSampleRate,
                           std::vector<float>& output) {
    if (inputSampleRate == outputSampleRate) {
        output = input;
        return;
    }
    double ratio = static_cast<double>(outputSampleRate) / inputSampleRate;
    size_t outputLen = static_cast<size_t>(input.size() * ratio);
    output.resize(outputLen);

    for (size_t i = 0; i < outputLen; ++i) {
        double pos = i / ratio;
        size_t idx = static_cast<size_t>(pos);
        double frac = pos - idx;
        double a = (idx < input.size()) ? input[idx] : 0.0;
        double b = (idx + 1 < input.size()) ? input[idx + 1] : a;
        output[i] = static_cast<float>(a + (b - a) * frac);
    }
}

DenoiseResult Denoiser::process(std::vector<float>& audio, int sampleRate) {
    DenoiseResult result;
    auto t0 = std::chrono::high_resolution_clock::now();

    if (audio.empty()) {
        result.success = false;
        result.errorMessage = "Empty audio buffer";
        return result;
    }

    if (!impl_->bridgeHandle) {
        result.success = false;
        result.errorMessage = "Denoiser not initialized (bridge handle null)";
        return result;
    }

    if (impl_->cfg.verbose) {
        fprintf(stderr, "[denoiser] Processing %zu samples at %d Hz\n",
                audio.size(), sampleRate);
    }

    float inputFloor = impl_->estimator.estimateNoiseFloorDb(
        audio.data(), audio.size(), sampleRate);
    result.noiseFloorDbfs = inputFloor;

    int processRate = impl_->nativeSampleRate;
    std::vector<float> processBuf;
    bool needsResample = (sampleRate != processRate);

    if (needsResample) {
        resampleBuffer(audio, sampleRate, processRate, processBuf);
    } else {
        processBuf = audio;
    }

    size_t frameSize = impl_->frameSize;
    if (frameSize == 0) {
        frameSize = 1024;
    }

    size_t numProcessed = 0;
    for (size_t offset = 0; offset + frameSize <= processBuf.size();
         offset += frameSize) {
        int ret = df_bridge_process(
            impl_->bridgeHandle,
            processBuf.data() + offset,
            frameSize);
        if (ret != 0) {
            result.success = false;
            result.errorMessage = "df_bridge_process failed";
            return result;
        }
        ++numProcessed;
    }

    size_t remaining = processBuf.size() % frameSize;
    if (remaining > 0) {
        size_t offset = processBuf.size() - remaining;
        int ret = df_bridge_process(
            impl_->bridgeHandle,
            processBuf.data() + offset,
            remaining);
        if (ret == 0) ++numProcessed;
    }

    result.framesProcessed = numProcessed;

    if (needsResample) {
        std::vector<float> resampledBack;
        resampleBuffer(processBuf, processRate, sampleRate, resampledBack);
        resampledBack.resize(audio.size());
        std::copy(resampledBack.begin(), resampledBack.end(), audio.begin());
    } else {
        std::copy(processBuf.begin(), processBuf.end(), audio.begin());
    }

    float outputFloor = impl_->estimator.estimateNoiseFloorDb(
        audio.data(), audio.size(), sampleRate);

    float inputSnr = impl_->estimator.estimateSnrDb(
        audio.data(), audio.size(), sampleRate);

    impl_->estimator = NoiseEstimator{};

    std::vector<float> original(audio);
    if (!needsResample && result.framesProcessed > 0) {
        float outputSnr = impl_->estimator.estimateSnrDb(
            audio.data(), audio.size(), sampleRate);
        result.outputSnrEstDb = outputSnr;
        result.inputSnrEstDb = inputSnr;
    } else {
        result.inputSnrEstDb = inputSnr;
        result.outputSnrEstDb = inputSnr + (inputFloor - outputFloor);
    }

    result.noiseFloorDbfs = outputFloor;
    result.success = true;

    auto t1 = std::chrono::high_resolution_clock::now();
    result.processingTimeMs =
        std::chrono::duration<float, std::milli>(t1 - t0).count();

    return result;
}

DenoiseResult Denoiser::processFile(const std::string& inputPath,
                                    const std::string& outputPath) {
    DenoiseResult result;

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

    if (impl_->cfg.verbose) {
        fprintf(stderr, "[denoiser] Input: %s\n", inputPath.c_str());
        fprintf(stderr, "[denoiser]   Channels:    %d\n", channels);
        fprintf(stderr, "[denoiser]   Sample rate: %d Hz\n", sampleRate);
        fprintf(stderr, "[denoiser]   Frames:      %lld\n",
                static_cast<long long>(totalFrames));
    }

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

    std::vector<std::vector<float>> channelBufs(channels);
    for (int ch = 0; ch < channels; ++ch) {
        channelBufs[ch].resize(totalFrames);
        for (sf_count_t f = 0; f < totalFrames; ++f) {
            channelBufs[ch][f] = interleavedBuf[f * channels + ch];
        }
    }

    DenoiseResult aggregateResult;
    for (int ch = 0; ch < channels; ++ch) {
        if (impl_->cfg.verbose) {
            fprintf(stderr, "[denoiser] Processing channel %d/%d...\n",
                    ch + 1, channels);
        }

        auto chResult = process(channelBufs[ch], sampleRate);

        if (ch == 0) {
            aggregateResult = chResult;
        } else {
            aggregateResult.framesProcessed += chResult.framesProcessed;
        }
    }

    for (int ch = 0; ch < channels; ++ch) {
        for (sf_count_t f = 0; f < totalFrames; ++f) {
            interleavedBuf[f * channels + ch] = channelBufs[ch][f];
        }
    }

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

    if (impl_->cfg.verbose) {
        fprintf(stderr, "[denoiser] Output written: %s\n",
                outputPath.c_str());
    }

    return aggregateResult;
}

}  // namespace faurge
