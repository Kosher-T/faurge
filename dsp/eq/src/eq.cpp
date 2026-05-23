#include "faurge/eq.hpp"
#include "faurge/eq_metrics.hpp"

#include <sndfile.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace faurge {

Equalizer::Equalizer(const EqConfig& config) : cfg_(config) {}

float Equalizer::dbToLinear(float db) {
    return std::pow(10.0f, db / 20.0f);
}

float Equalizer::peakDb(const float* audio, size_t n) {
    float peak = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float absVal = std::fabs(audio[i]);
        if (absVal > peak) peak = absVal;
    }
    if (peak < 1e-30f) return -120.0f;
    return 20.0f * std::log10(peak);
}

float Equalizer::rmsDb(const float* audio, size_t n) {
    if (n == 0) return -120.0f;
    float sumSq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sumSq += audio[i] * audio[i];
    }
    float rms = std::sqrt(sumSq / static_cast<float>(n));
    if (rms < 1e-30f) return -120.0f;
    return 20.0f * std::log10(rms);
}

EqResult Equalizer::processImpl(std::vector<float>& audio, int sampleRate,
                                const EqConfig& config, int channel) {
    EqResult result;

    if (audio.empty()) {
        result.success = false;
        result.errorMessage = "Empty audio buffer";
        return result;
    }

    size_t numSamples = audio.size();

    result.inputPeakDb = peakDb(audio.data(), numSamples);
    result.inputRmsDb = rmsDb(audio.data(), numSamples);

    std::vector<float> output(numSamples, 0.0f);
    for (int b = 0; b < NUM_EQ_BANDS; ++b) {
        const auto& band = config.bands[b];
        float bandGain = band.gain_db;
        if (channel >= 0 && band.stereo_skew_db != 0.0f) {
            float halfSkew = band.stereo_skew_db * 0.5f;
            bandGain += (channel == 0) ? halfSkew : -halfSkew;
        }
        bands_[b].reset();
        bands_[b].setFilter(band.filter_type, band.freq_hz,
                            bandGain, band.q, sampleRate);
        std::vector<float> bandBuf(audio);
        bands_[b].process(bandBuf);
        for (size_t i = 0; i < numSamples; ++i) {
            output[i] += bandBuf[i];
        }
    }
    for (size_t i = 0; i < numSamples; ++i) {
        output[i] -= static_cast<float>(NUM_EQ_BANDS - 1) * audio[i];
    }
    audio.swap(output);

    result.outputPeakDb = peakDb(audio.data(), numSamples);
    result.outputRmsDb = rmsDb(audio.data(), numSamples);
    result.framesProcessed = numSamples;
    result.success = true;

    return result;
}

EqResult Equalizer::process(std::vector<float>& audio, int sampleRate) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = processImpl(audio, sampleRate, cfg_);
    auto t1 = std::chrono::high_resolution_clock::now();
    result.processingTimeMs =
        std::chrono::duration<float, std::milli>(t1 - t0).count();
    return result;
}

EqResult Equalizer::process(std::vector<float>& audio, int sampleRate,
                            const EqConfig& config) {
    cfg_ = config;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = processImpl(audio, sampleRate, cfg_);
    auto t1 = std::chrono::high_resolution_clock::now();
    result.processingTimeMs =
        std::chrono::duration<float, std::milli>(t1 - t0).count();
    return result;
}

EqResult Equalizer::process(std::vector<float>& audio, int sampleRate,
                            const EqConfig& config, int channel) {
    cfg_ = config;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = processImpl(audio, sampleRate, cfg_, channel);
    auto t1 = std::chrono::high_resolution_clock::now();
    result.processingTimeMs =
        std::chrono::duration<float, std::milli>(t1 - t0).count();
    return result;
}

EqResult Equalizer::processFile(const std::string& inputPath,
                                const std::string& outputPath) {
    EqResult result;

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
        fprintf(stderr, "[eq] Input: %s\n", inputPath.c_str());
        fprintf(stderr, "[eq]   Channels:    %d\n", channels);
        fprintf(stderr, "[eq]   Sample rate: %d Hz\n", sampleRate);
        fprintf(stderr, "[eq]   Frames:      %lld\n",
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

    for (int ch = 0; ch < channels; ++ch) {
        if (cfg_.verbose) {
            fprintf(stderr, "[eq] Processing channel %d/%d...\n",
                    ch + 1, channels);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        auto chResult = processImpl(channelBufs[ch], sampleRate, cfg_, ch);
        auto t1 = std::chrono::high_resolution_clock::now();
        chResult.processingTimeMs =
            std::chrono::duration<float, std::milli>(t1 - t0).count();

        if (ch == 0) {
            result = chResult;
        } else {
            result.framesProcessed += chResult.framesProcessed;
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
        result.success = false;
        result.errorMessage = std::string("Failed to open output: ")
                            + sf_strerror(nullptr);
        return result;
    }

    sf_writef_float(outFile, interleavedBuf.data(), totalFrames);
    sf_close(outFile);

    if (cfg_.verbose) {
        fprintf(stderr, "[eq] Output written: %s\n", outputPath.c_str());
    }

    return result;
}

}
