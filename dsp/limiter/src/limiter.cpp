#include "faurge/limiter.hpp"
#include "faurge/limiter_metrics.hpp"

#include <sndfile.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace faurge {

Limiter::Limiter(const LimiterConfig& config) : cfg_(config) {}

float Limiter::dbToLinear(float db) {
    return std::pow(10.0f, db / 20.0f);
}

float Limiter::linearToDb(float linear) {
    if (linear < 1e-30f) return -120.0f;
    return 20.0f * std::log10(linear);
}

float Limiter::peakDb(const float* audio, size_t n) {
    float peak = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float absVal = std::fabs(audio[i]);
        if (absVal > peak) peak = absVal;
    }
    if (peak < 1e-30f) return -120.0f;
    return 20.0f * std::log10(peak);
}

float Limiter::rmsDb(const float* audio, size_t n) {
    if (n == 0) return -120.0f;
    float sumSq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sumSq += audio[i] * audio[i];
    }
    float rms = std::sqrt(sumSq / static_cast<float>(n));
    if (rms < 1e-30f) return -120.0f;
    return 20.0f * std::log10(rms);
}

LimiterResult Limiter::processImpl(std::vector<float>& audio, int sampleRate,
                                    const LimiterConfig& config) {
    LimiterResult result;

    if (audio.empty() || sampleRate <= 0) {
        result.success = false;
        result.errorMessage = "Empty audio buffer or invalid sample rate";
        return result;
    }

    size_t n = audio.size();

    result.inputPeakDb = peakDb(audio.data(), n);
    result.inputRmsDb = rmsDb(audio.data(), n);

    predictor_.reset();
    smoothedGrDb_ = 0.0f;

    int lookaheadSamples = static_cast<int>(
        config.lookahead_ms * static_cast<float>(sampleRate) * 0.001f);
    predictor_.configure(lookaheadSamples);

    float ceilingLin = dbToLinear(config.ceiling_db);
    float releaseAlpha = (config.release_ms <= 0.0f) ? 1.0f :
        (1.0f - std::exp(-2.2f / (config.release_ms *
                                   static_cast<float>(sampleRate) * 0.001f)));
    releaseAlpha = std::min(1.0f, releaseAlpha);

    bool softClip = (config.clip_mode == ClipMode::soft);
    float tanhScale = 1.313f;

    float sumGrAbs = 0.0f;
    float maxGr = 0.0f;
    int clippedSamples = 0;

    for (size_t i = 0; i < n; ++i) {
        float delayedSample, predictedPeak;
        predictor_.processSample(audio[i], delayedSample, predictedPeak);

        float targetGrDb;
        if (predictedPeak <= ceilingLin) {
            targetGrDb = 0.0f;
        } else {
            targetGrDb = linearToDb(ceilingLin / predictedPeak);
        }

        if (targetGrDb < smoothedGrDb_) {
            smoothedGrDb_ = targetGrDb;
        } else {
            smoothedGrDb_ += releaseAlpha * (targetGrDb - smoothedGrDb_);
        }

        float gainLin = dbToLinear(smoothedGrDb_);
        float outSample = delayedSample * gainLin;

        if (softClip) {
            float scaled = outSample / ceilingLin;
            outSample = ceilingLin * tanhScale * std::tanh(scaled / tanhScale);
        } else {
            if (outSample > ceilingLin) {
                outSample = ceilingLin;
                ++clippedSamples;
            } else if (outSample < -ceilingLin) {
                outSample = -ceilingLin;
                ++clippedSamples;
            }
        }

        audio[i] = outSample;

        float absGr = std::fabs(smoothedGrDb_);
        sumGrAbs += absGr;
        if (absGr > maxGr) maxGr = absGr;
    }

    result.outputPeakDb = peakDb(audio.data(), n);
    result.outputRmsDb = rmsDb(audio.data(), n);
    result.maxGainReductionDb = maxGr;
    result.avgGainReductionDb = sumGrAbs / static_cast<float>(n);
    result.clippedSamples = clippedSamples;
    result.framesProcessed = n;
    result.success = true;

    return result;
}

LimiterResult Limiter::process(std::vector<float>& audio, int sampleRate) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = processImpl(audio, sampleRate, cfg_);
    auto t1 = std::chrono::high_resolution_clock::now();
    result.processingTimeMs =
        std::chrono::duration<float, std::milli>(t1 - t0).count();
    return result;
}

LimiterResult Limiter::process(std::vector<float>& audio, int sampleRate,
                                const LimiterConfig& config) {
    cfg_ = config;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = processImpl(audio, sampleRate, cfg_);
    auto t1 = std::chrono::high_resolution_clock::now();
    result.processingTimeMs =
        std::chrono::duration<float, std::milli>(t1 - t0).count();
    return result;
}

LimiterResult Limiter::processFile(const std::string& inputPath,
                                    const std::string& outputPath) {
    LimiterResult result;

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
        fprintf(stderr, "[limit] Input: %s\n", inputPath.c_str());
        fprintf(stderr, "[limit]   Channels:    %d\n", channels);
        fprintf(stderr, "[limit]   Sample rate: %d Hz\n", sampleRate);
        fprintf(stderr, "[limit]   Frames:      %lld\n",
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

    if (channels == 2 && cfg_.stereo_link > 0.0f) {
        if (cfg_.verbose) {
            fprintf(stderr, "[limit] Stereo link enabled (%.2f)\n", cfg_.stereo_link);
        }

        predictor_.reset();
        smoothedGrDb_ = 0.0f;

        int lookaheadSamples = static_cast<int>(
            cfg_.lookahead_ms * static_cast<float>(sampleRate) * 0.001f);
        predictor_.configure(lookaheadSamples);

        float ceilingLin = dbToLinear(cfg_.ceiling_db);
        float releaseAlpha = (cfg_.release_ms <= 0.0f) ? 1.0f :
            (1.0f - std::exp(-2.2f / (cfg_.release_ms *
                                       static_cast<float>(sampleRate) * 0.001f)));
        releaseAlpha = std::min(1.0f, releaseAlpha);

        bool softClip = (cfg_.clip_mode == ClipMode::soft);
        float tanhScale = 1.313f;

        float sumGrAbs = 0.0f;
        float maxGr = 0.0f;
        int clippedSamples = 0;

        auto t0 = std::chrono::high_resolution_clock::now();

        for (sf_count_t f = 0; f < totalFrames; ++f) {
            float inL = channelBufs[0][f];
            float inR = channelBufs[1][f];

            float delayedL, delayedR;
            float peakL, peakR;
            predictor_.processSample(inL, delayedL, peakL);
            predictor_.processSample(inR, delayedR, peakR);

            float predictedPeak = peakL * cfg_.stereo_link
                                + peakR * (1.0f - cfg_.stereo_link);

            float targetGrDb;
            if (predictedPeak <= ceilingLin) {
                targetGrDb = 0.0f;
            } else {
                targetGrDb = linearToDb(ceilingLin / predictedPeak);
            }

            if (targetGrDb < smoothedGrDb_) {
                smoothedGrDb_ = targetGrDb;
            } else {
                smoothedGrDb_ += releaseAlpha * (targetGrDb - smoothedGrDb_);
            }

            float gainLin = dbToLinear(smoothedGrDb_);

            for (int ch = 0; ch < 2; ++ch) {
                float outSample = channelBufs[ch][f] * gainLin;

                if (softClip) {
                    float scaled = outSample / ceilingLin;
                    outSample = ceilingLin * tanhScale * std::tanh(scaled / tanhScale);
                } else {
                    if (outSample > ceilingLin) {
                        outSample = ceilingLin;
                        ++clippedSamples;
                    } else if (outSample < -ceilingLin) {
                        outSample = -ceilingLin;
                        ++clippedSamples;
                    }
                }

                channelBufs[ch][f] = outSample;
            }

            float absGr = std::fabs(smoothedGrDb_);
            sumGrAbs += absGr;
            if (absGr > maxGr) maxGr = absGr;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        result.processingTimeMs =
            std::chrono::duration<float, std::milli>(t1 - t0).count();

        result.inputPeakDb = peakDb(interleavedBuf.data(), totalSamples);
        result.inputRmsDb = rmsDb(interleavedBuf.data(), totalSamples);

        float outPeak = 0.0f;
        float outSumSq = 0.0f;
        for (int ch = 0; ch < 2; ++ch) {
            for (sf_count_t f = 0; f < totalFrames; ++f) {
                float absVal = std::fabs(channelBufs[ch][f]);
                if (absVal > outPeak) outPeak = absVal;
                outSumSq += channelBufs[ch][f] * channelBufs[ch][f];
            }
        }
        result.outputPeakDb = linearToDb(outPeak);
        float outRms = std::sqrt(outSumSq / static_cast<float>(totalFrames * 2));
        result.outputRmsDb = linearToDb(outRms);

        result.maxGainReductionDb = maxGr;
        result.avgGainReductionDb = sumGrAbs / static_cast<float>(totalFrames);
        result.clippedSamples = clippedSamples;
        result.framesProcessed = static_cast<size_t>(totalFrames) * 2;
        result.success = true;
    } else {
        for (int ch = 0; ch < channels; ++ch) {
            if (cfg_.verbose) {
                fprintf(stderr, "[limit] Processing channel %d/%d...\n",
                        ch + 1, channels);
            }

            if (channels == 2 && cfg_.stereo_link > 0.0f && ch > 0) {
                continue;
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

        if (channels == 2 && cfg_.stereo_link > 0.0f) {
            channelBufs[1] = channelBufs[0];
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
        fprintf(stderr, "[limit] Output written: %s\n", outputPath.c_str());
    }

    return result;
}

} // namespace faurge
