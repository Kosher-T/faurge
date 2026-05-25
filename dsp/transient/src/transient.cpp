#include "faurge/transient.hpp"
#include "faurge/transient_metrics.hpp"

#include <sndfile.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace faurge {

TransientShaper::TransientShaper(const TransientConfig& config) : cfg_(config) {}

float TransientShaper::dbToLinear(float db) {
    return std::pow(10.0f, db / 20.0f);
}

float TransientShaper::linearToDb(float linear) {
    if (linear < 1e-30f) return -120.0f;
    return 20.0f * std::log10(linear);
}

float TransientShaper::peakDb(const float* audio, size_t n) {
    float peak = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float absVal = std::fabs(audio[i]);
        if (absVal > peak) peak = absVal;
    }
    if (peak < 1e-30f) return -120.0f;
    return 20.0f * std::log10(peak);
}

float TransientShaper::rmsDb(const float* audio, size_t n) {
    if (n == 0) return -120.0f;
    float sumSq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sumSq += audio[i] * audio[i];
    }
    float rms = std::sqrt(sumSq / static_cast<float>(n));
    if (rms < 1e-30f) return -120.0f;
    return 20.0f * std::log10(rms);
}

TransientResult TransientShaper::processImpl(std::vector<float>& audio,
                                              int sampleRate,
                                              const TransientConfig& config) {
    TransientResult result;

    if (audio.empty() || sampleRate <= 0) {
        result.success = false;
        result.errorMessage = "Empty audio buffer or invalid sample rate";
        return result;
    }

    size_t n = audio.size();

    result.inputPeakDb = peakDb(audio.data(), n);
    result.inputRmsDb = rmsDb(audio.data(), n);

    splitter_.reset();

    std::vector<float> factors(n);

    splitter_.processFactors(audio.data(), n,
                              config.attack_time_ms, config.release_time_ms,
                              config.sensitivity_db, sampleRate, factors.data());

    float attLin = dbToLinear(config.attack_gain_db);
    float susLin = dbToLinear(config.sustain_gain_db);

    std::vector<float> processed(n);
    TransientSplitter::applyGain(audio.data(), processed.data(), n,
                                  factors.data(), attLin, susLin, config.mix);

    for (size_t i = 0; i < n; ++i) {
        audio[i] = processed[i];
    }

    result.outputPeakDb = peakDb(audio.data(), n);
    result.outputRmsDb = rmsDb(audio.data(), n);
    result.peakToRmsDb = result.outputPeakDb - result.outputRmsDb;

    {
        float sumGainAtt = 0.0f;
        float sumGainSus = 0.0f;
        int countAtt = 0;
        int countSus = 0;

        for (size_t i = 0; i < n; ++i) {
            float f = factors[i];
            float gain;
            if (f > 0.0f) {
                gain = 1.0f + (attLin - 1.0f) * f;
            } else if (f < 0.0f) {
                gain = 1.0f + (susLin - 1.0f) * (-f);
            } else {
                gain = 1.0f;
            }
            float gainDb = linearToDb(gain);
            if (f > 0.5f) {
                sumGainAtt += gainDb;
                ++countAtt;
            } else if (f < -0.5f) {
                sumGainSus += gainDb;
                ++countSus;
            }
        }

        result.avgAttackDb = (countAtt > 0) ? (sumGainAtt / static_cast<float>(countAtt)) : 0.0f;
        result.avgSustainDb = (countSus > 0) ? (sumGainSus / static_cast<float>(countSus)) : 0.0f;

        if (n > 0 && countAtt == 0 && countSus == 0) {
            result.avgAttackDb = 0.0f;
            result.avgSustainDb = 0.0f;
        }
    }

    result.framesProcessed = n;
    result.success = true;

    return result;
}

TransientResult TransientShaper::process(std::vector<float>& audio, int sampleRate) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = processImpl(audio, sampleRate, cfg_);
    auto t1 = std::chrono::high_resolution_clock::now();
    result.processingTimeMs =
        std::chrono::duration<float, std::milli>(t1 - t0).count();
    return result;
}

TransientResult TransientShaper::process(std::vector<float>& audio, int sampleRate,
                                          const TransientConfig& config) {
    cfg_ = config;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = processImpl(audio, sampleRate, cfg_);
    auto t1 = std::chrono::high_resolution_clock::now();
    result.processingTimeMs =
        std::chrono::duration<float, std::milli>(t1 - t0).count();
    return result;
}

TransientResult TransientShaper::processFile(const std::string& inputPath,
                                              const std::string& outputPath) {
    TransientResult result;

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
        fprintf(stderr, "[transient] Input: %s\n", inputPath.c_str());
        fprintf(stderr, "[transient]   Channels:    %d\n", channels);
        fprintf(stderr, "[transient]   Sample rate: %d Hz\n", sampleRate);
        fprintf(stderr, "[transient]   Frames:      %lld\n",
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
            fprintf(stderr, "[transient] Processing channel %d/%d...\n",
                    ch + 1, channels);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        auto chResult = processImpl(channelBufs[ch], sampleRate, cfg_);
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
        fprintf(stderr, "[transient] Output written: %s\n", outputPath.c_str());
    }

    return result;
}

} // namespace faurge
