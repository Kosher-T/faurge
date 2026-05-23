#include "faurge/esser.hpp"
#include "faurge/esser_metrics.hpp"

#include <sndfile.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace faurge {

Esser::Esser(const EsserConfig& config) : cfg_(config) {}

float Esser::dbToLinear(float db) {
    return std::pow(10.0f, db / 20.0f);
}

float Esser::linearToDb(float linear) {
    if (linear < 1e-30f) return -120.0f;
    return 20.0f * std::log10(linear);
}

float Esser::peakDb(const float* audio, size_t n) {
    float peak = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float absVal = std::fabs(audio[i]);
        if (absVal > peak) peak = absVal;
    }
    if (peak < 1e-30f) return -120.0f;
    return 20.0f * std::log10(peak);
}

float Esser::rmsDb(const float* audio, size_t n) {
    if (n == 0) return -120.0f;
    float sumSq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sumSq += audio[i] * audio[i];
    }
    float rms = std::sqrt(sumSq / static_cast<float>(n));
    if (rms < 1e-30f) return -120.0f;
    return 20.0f * std::log10(rms);
}

static float rcAlpha(float tauMs, int sampleRate) {
    if (tauMs <= 0.0f) return 1.0f;
    return 1.0f - std::exp(-2.2f / (tauMs * static_cast<float>(sampleRate) * 0.001f));
}

static float computeGainReduction(float envelopeDb, float thresholdDb,
                                   float ratio) {
    float overshoot = envelopeDb - thresholdDb;

    if (ratio >= 1.0f) {
        if (overshoot <= 0.0f) return 0.0f;
        return -overshoot * (ratio - 1.0f) / ratio;
    } else {
        if (overshoot >= 0.0f) return 0.0f;
        return overshoot * (1.0f - ratio);
    }
}

EsserResult Esser::processImpl(std::vector<float>& audio, int sampleRate,
                                const EsserConfig& config) {
    EsserResult result;

    if (audio.empty()) {
        result.success = false;
        result.errorMessage = "Empty audio buffer";
        return result;
    }

    size_t numSamples = audio.size();
    sampleRate_ = sampleRate;

    result.inputPeakDb = peakDb(audio.data(), numSamples);
    result.inputRmsDb = rmsDb(audio.data(), numSamples);

    float q = config.center_freq_hz / std::max(config.bandwidth_hz, 50.0f);
    float effectiveAttackMs = std::max(config.attack_ms,
                                       2.0f * q / config.center_freq_hz * 1000.0f);

    float attackAlpha = rcAlpha(effectiveAttackMs, sampleRate);
    float releaseAlpha = rcAlpha(config.release_ms, sampleRate);
    float attackAlphaGr = rcAlpha(effectiveAttackMs, sampleRate);
    float releaseAlphaGr = rcAlpha(config.release_ms, sampleRate);

    detector_.configure(config.center_freq_hz, config.bandwidth_hz, sampleRate);

    size_t warmupSamples = std::max(static_cast<size_t>(sampleRate * 0.005f),
                                     size_t(1));
    for (size_t i = 0; i < warmupSamples; ++i) {
        detector_.processSample(0.0f);
    }

    envelope_ = 0.0f;
    smoothedGainDb_ = 0.0f;

    float maxGr = 0.0f;
    float sumActiveGr = 0.0f;
    int activeFrames = 0;

    for (size_t i = 0; i < numSamples; ++i) {
        float inputSample = audio[i];

        float detected = detector_.processSample(inputSample);

        float envIn = std::sqrt(std::max(1e-30f, detected * detected));
        float alpha = (envIn >= envelope_) ? attackAlpha : releaseAlpha;
        envelope_ += alpha * (envIn - envelope_);

        float envDb = linearToDb(envelope_);
        float targetGr = computeGainReduction(envDb, config.threshold_db,
                                               config.ratio);

        float alphaGr = (targetGr < smoothedGainDb_)
                            ? attackAlphaGr
                            : releaseAlphaGr;
        smoothedGainDb_ += alphaGr * (targetGr - smoothedGainDb_);

        float gainLin = dbToLinear(smoothedGainDb_);

        audio[i] = inputSample * gainLin;

        float absGr = std::fabs(smoothedGainDb_);
        if (absGr > maxGr) maxGr = absGr;
        if (absGr > 0.1f) {
            sumActiveGr += absGr;
            ++activeFrames;
        }
    }

    result.outputPeakDb = peakDb(audio.data(), numSamples);
    result.outputRmsDb = rmsDb(audio.data(), numSamples);
    result.maxGainReductionDb = maxGr;
    result.avgActiveGainReductionDb = (activeFrames > 0)
                                          ? (sumActiveGr / activeFrames)
                                          : 0.0f;
    result.sibilantFrames = activeFrames;
    result.framesProcessed = numSamples;
    result.success = true;

    return result;
}

EsserResult Esser::process(std::vector<float>& audio, int sampleRate) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = processImpl(audio, sampleRate, cfg_);
    auto t1 = std::chrono::high_resolution_clock::now();
    result.processingTimeMs =
        std::chrono::duration<float, std::milli>(t1 - t0).count();
    return result;
}

EsserResult Esser::process(std::vector<float>& audio, int sampleRate,
                            const EsserConfig& config) {
    cfg_ = config;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = processImpl(audio, sampleRate, cfg_);
    auto t1 = std::chrono::high_resolution_clock::now();
    result.processingTimeMs =
        std::chrono::duration<float, std::milli>(t1 - t0).count();
    return result;
}

EsserResult Esser::processFile(const std::string& inputPath,
                                const std::string& outputPath) {
    EsserResult result;

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
        fprintf(stderr, "[esser] Input: %s\n", inputPath.c_str());
        fprintf(stderr, "[esser]   Channels:    %d\n", channels);
        fprintf(stderr, "[esser]   Sample rate: %d Hz\n", sampleRate);
        fprintf(stderr, "[esser]   Frames:      %lld\n",
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

    result.inputPeakDb = peakDb(interleavedBuf.data(), totalSamples);
    result.inputRmsDb = rmsDb(interleavedBuf.data(), totalSamples);

    std::vector<std::vector<float>> channelBufs(channels);
    for (int ch = 0; ch < channels; ++ch) {
        channelBufs[ch].resize(totalFrames);
        for (sf_count_t f = 0; f < totalFrames; ++f) {
            channelBufs[ch][f] = interleavedBuf[f * channels + ch];
        }
    }

    float q = cfg_.center_freq_hz / std::max(cfg_.bandwidth_hz, 50.0f);
    float effectiveAttackMs = std::max(cfg_.attack_ms,
                                       2.0f * q / cfg_.center_freq_hz * 1000.0f);
    float attackAlpha = rcAlpha(effectiveAttackMs, sampleRate);
    float releaseAlpha = rcAlpha(cfg_.release_ms, sampleRate);
    float attackAlphaGr = rcAlpha(effectiveAttackMs, sampleRate);
    float releaseAlphaGr = rcAlpha(cfg_.release_ms, sampleRate);

    detector_.configure(cfg_.center_freq_hz, cfg_.bandwidth_hz, sampleRate);

    size_t warmupSamples = std::max(static_cast<size_t>(sampleRate * 0.005f),
                                     size_t(1));
    for (size_t i = 0; i < warmupSamples; ++i) {
        detector_.processSample(0.0f);
    }

    float totalSumActiveGr = 0.0f;
    int totalActiveFrames = 0;

    for (int ch = 0; ch < channels; ++ch) {
        if (cfg_.verbose) {
            fprintf(stderr, "[esser] Processing channel %d/%d...\n",
                    ch + 1, channels);
        }

        envelope_ = 0.0f;
        smoothedGainDb_ = 0.0f;
        float maxGr = 0.0f;
        float sumActiveGr = 0.0f;
        int activeFrames = 0;

        auto& buf = channelBufs[ch];
        size_t n = buf.size();

        for (size_t i = 0; i < n; ++i) {
            float inputSample = buf[i];
            float detected = detector_.processSample(inputSample);

            float envIn = std::sqrt(std::max(1e-30f, detected * detected));
            float alpha = (envIn >= envelope_) ? attackAlpha : releaseAlpha;
            envelope_ += alpha * (envIn - envelope_);

            float envDb = linearToDb(envelope_);
            float targetGr = computeGainReduction(envDb, cfg_.threshold_db,
                                                   cfg_.ratio);

            float alphaGr = (targetGr < smoothedGainDb_)
                                ? attackAlphaGr
                                : releaseAlphaGr;
            smoothedGainDb_ += alphaGr * (targetGr - smoothedGainDb_);

            float gainLin = dbToLinear(smoothedGainDb_);
            buf[i] = inputSample * gainLin;

            float absGr = std::fabs(smoothedGainDb_);
            if (absGr > maxGr) maxGr = absGr;
            if (absGr > 0.1f) {
                sumActiveGr += absGr;
                ++activeFrames;
            }
        }

        result.maxGainReductionDb = std::max(result.maxGainReductionDb, maxGr);
        result.sibilantFrames += activeFrames;
        result.framesProcessed += n;
        totalSumActiveGr += sumActiveGr;
        totalActiveFrames += activeFrames;
    }

    result.avgActiveGainReductionDb = (totalActiveFrames > 0)
        ? (totalSumActiveGr / totalActiveFrames) : 0.0f;

    for (int ch = 0; ch < channels; ++ch) {
        for (sf_count_t f = 0; f < totalFrames; ++f) {
            interleavedBuf[f * channels + ch] = channelBufs[ch][f];
        }
    }

    result.outputPeakDb = peakDb(interleavedBuf.data(), totalSamples);
    result.outputRmsDb = rmsDb(interleavedBuf.data(), totalSamples);
    result.success = true;

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
        fprintf(stderr, "[esser] Output written: %s\n", outputPath.c_str());
    }

    return result;
}

} // namespace faurge
