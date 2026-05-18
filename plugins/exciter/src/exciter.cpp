#include "faurge/exciter.hpp"
#include "faurge/crossover_filter.hpp"
#include "faurge/high_band.hpp"
#include "faurge/low_band.hpp"
#include "faurge/exciter_metrics.hpp"

#include <sndfile.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace faurge {

Exciter::Exciter(const ExciterConfig& config) : cfg_(config) {}

float Exciter::dbToLinear(float db) {
    return std::pow(10.0f, db / 20.0f);
}

float Exciter::peakDb(const float* audio, size_t n) {
    float peak = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float absVal = std::fabs(audio[i]);
        if (absVal > peak) peak = absVal;
    }
    if (peak < 1e-30f) return -120.0f;
    return 20.0f * std::log10(peak);
}

float Exciter::rmsDb(const float* audio, size_t n) {
    if (n == 0) return -120.0f;
    float sumSq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sumSq += audio[i] * audio[i];
    }
    float rms = std::sqrt(sumSq / static_cast<float>(n));
    if (rms < 1e-30f) return -120.0f;
    return 20.0f * std::log10(rms);
}

float Exciter::bandEnergyDb(const float* audio, size_t n) {
    if (n == 0) return -120.0f;
    float sumSq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sumSq += audio[i] * audio[i];
    }
    float energy = sumSq / static_cast<float>(n);
    if (energy < 1e-30f) return -120.0f;
    return 10.0f * std::log10(energy);
}

ExciterResult Exciter::process(std::vector<float>& audio, int sampleRate) {
    ExciterResult result;
    auto t0 = std::chrono::high_resolution_clock::now();

    if (audio.empty()) {
        result.success = false;
        result.errorMessage = "Empty audio buffer";
        return result;
    }

    size_t numSamples = audio.size();

    result.inputPeakDb = peakDb(audio.data(), numSamples);
    result.inputRmsDb = rmsDb(audio.data(), numSamples);

    std::vector<float> highBuf(numSamples);
    std::vector<float> lowBuf(numSamples);
    std::vector<float> highOut(numSamples);
    std::vector<float> lowOut(numSamples);

    CrossoverFilter highCross;
    CrossoverFilter lowCross;

    highCross.process(audio.data(), nullptr, highBuf.data(),
                      numSamples, sampleRate, cfg_.highCrossoverHz);
    lowCross.process(audio.data(), lowBuf.data(), nullptr,
                     numSamples, sampleRate, cfg_.lowCrossoverHz);

    if (cfg_.highEnable) {
        HighBand highBand;
        highBand.process(highBuf.data(), highOut.data(),
                         numSamples, sampleRate, cfg_.highDriveDb);
    } else {
        std::memcpy(highOut.data(), highBuf.data(), numSamples * sizeof(float));
    }

    if (cfg_.lowEnable) {
        LowBand lowBand;
        lowBand.process(lowBuf.data(), lowOut.data(),
                        numSamples, sampleRate,
                        cfg_.lowDriveDb, cfg_.lowSubLevel);
    } else {
        std::memcpy(lowOut.data(), lowBuf.data(), numSamples * sizeof(float));
    }

    for (size_t i = 0; i < numSamples; ++i) {
        float highDelta = highOut[i] - highBuf[i];
        float lowDelta = lowOut[i] - lowBuf[i];
        float s = audio[i] + cfg_.highMix * highDelta + cfg_.lowMix * lowDelta;
        s *= cfg_.masterVolume;
        audio[i] = s;
    }

    result.outputPeakDb = peakDb(audio.data(), numSamples);
    result.outputRmsDb = rmsDb(audio.data(), numSamples);
    result.highBandEnergyDb = bandEnergyDb(highBuf.data(), numSamples);
    result.lowBandEnergyDb = bandEnergyDb(lowBuf.data(), numSamples);
    result.framesProcessed = numSamples;
    result.success = true;

    auto t1 = std::chrono::high_resolution_clock::now();
    result.processingTimeMs =
        std::chrono::duration<float, std::milli>(t1 - t0).count();

    return result;
}

ExciterResult Exciter::processFile(const std::string& inputPath,
                                   const std::string& outputPath) {
    ExciterResult result;

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
        fprintf(stderr, "[exciter] Input: %s\n", inputPath.c_str());
        fprintf(stderr, "[exciter]   Channels:    %d\n", channels);
        fprintf(stderr, "[exciter]   Sample rate: %d Hz\n", sampleRate);
        fprintf(stderr, "[exciter]   Frames:      %lld\n",
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

    ExciterResult aggregateResult;
    for (int ch = 0; ch < channels; ++ch) {
        if (cfg_.verbose) {
            fprintf(stderr, "[exciter] Processing channel %d/%d...\n",
                    ch + 1, channels);
        }

        auto chResult = process(channelBufs[ch], sampleRate);

        if (ch == 0) {
            aggregateResult = chResult;
        } else {
            aggregateResult.highBandEnergyDb = std::max(
                aggregateResult.highBandEnergyDb, chResult.highBandEnergyDb);
            aggregateResult.lowBandEnergyDb = std::max(
                aggregateResult.lowBandEnergyDb, chResult.lowBandEnergyDb);
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

    if (cfg_.verbose) {
        fprintf(stderr, "[exciter] Output written: %s\n",
                outputPath.c_str());
    }

    return aggregateResult;
}

}
