#include "faurge/compressor.hpp"
#include "faurge/comp_metrics.hpp"

#include <sndfile.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace faurge {

namespace {

struct BiquadState {
    float x1 = 0.0f, x2 = 0.0f, y1 = 0.0f, y2 = 0.0f;
};

struct BiquadCoeffs {
    float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f;
    float a1 = 0.0f, a2 = 0.0f;
};

BiquadCoeffs designHp(float freq, int sampleRate) {
    const float PI = 3.14159265358979f;
    BiquadCoeffs c;
    float w0 = 2.0f * PI * freq / static_cast<float>(sampleRate);
    float cosW0 = std::cos(w0);
    float alpha = std::sin(w0) / 2.0f;
    c.b0 = (1.0f + cosW0) / 2.0f;
    c.b1 = -(1.0f + cosW0);
    c.b2 = (1.0f + cosW0) / 2.0f;
    float a0 = 1.0f + alpha;
    c.a1 = -2.0f * cosW0;
    c.a2 = 1.0f - alpha;
    c.b0 /= a0; c.b1 /= a0; c.b2 /= a0;
    c.a1 /= a0; c.a2 /= a0;
    return c;
}

BiquadCoeffs designLp(float freq, int sampleRate) {
    const float PI = 3.14159265358979f;
    BiquadCoeffs c;
    float w0 = 2.0f * PI * freq / static_cast<float>(sampleRate);
    float cosW0 = std::cos(w0);
    float alpha = std::sin(w0) / 2.0f;
    c.b0 = (1.0f - cosW0) / 2.0f;
    c.b1 = 1.0f - cosW0;
    c.b2 = (1.0f - cosW0) / 2.0f;
    float a0 = 1.0f + alpha;
    c.a1 = -2.0f * cosW0;
    c.a2 = 1.0f - alpha;
    c.b0 /= a0; c.b1 /= a0; c.b2 /= a0;
    c.a1 /= a0; c.a2 /= a0;
    return c;
}

float processBiquad(const BiquadCoeffs& c, BiquadState& s, float in) {
    float y = c.b0 * in + c.b1 * s.x1 + c.b2 * s.x2
            - c.a1 * s.y1 - c.a2 * s.y2;
    s.x2 = s.x1;
    s.x1 = in;
    s.y2 = s.y1;
    s.y1 = y;
    return y;
}

} // anonymous namespace

Compressor::Compressor(const CompConfig& config) : cfg_(config) {}

float Compressor::dbToLinear(float db) {
    return std::pow(10.0f, db / 20.0f);
}

float Compressor::linearToDb(float linear) {
    if (linear < 1e-30f) return -120.0f;
    return 20.0f * std::log10(linear);
}

float Compressor::peakDb(const float* audio, size_t n) {
    float peak = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float absVal = std::fabs(audio[i]);
        if (absVal > peak) peak = absVal;
    }
    if (peak < 1e-30f) return -120.0f;
    return 20.0f * std::log10(peak);
}

float Compressor::rmsDb(const float* audio, size_t n) {
    if (n == 0) return -120.0f;
    float sumSq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sumSq += audio[i] * audio[i];
    }
    float rms = std::sqrt(sumSq / static_cast<float>(n));
    if (rms < 1e-30f) return -120.0f;
    return 20.0f * std::log10(rms);
}

CompResult Compressor::processImpl(std::vector<float>& audio, int sampleRate,
                                    const CompConfig& config) {
    CompResult result;

    if (audio.empty() || sampleRate <= 0) {
        result.success = false;
        result.errorMessage = "Empty audio buffer or invalid sample rate";
        return result;
    }

    size_t n = audio.size();

    result.inputPeakDb = peakDb(audio.data(), n);
    result.inputRmsDb = rmsDb(audio.data(), n);

    envFollower_.reset();
    gainComputer_.reset();

    BiquadCoeffs scHp = designHp(config.sidechain_hp_hz, sampleRate);
    BiquadCoeffs scLp = designLp(config.sidechain_lp_hz, sampleRate);
    BiquadState scHpState, scLpState;

    int lookaheadSamples = static_cast<int>(
        config.lookahead_ms * static_cast<float>(sampleRate) * 0.001f);
    std::vector<float> envDelayBuf(lookaheadSamples, 0.0f);
    std::vector<float> audioDelayBuf(lookaheadSamples, 0.0f);
    int delayWriteIdx = 0;

    float sumGrAbs = 0.0f;
    float maxGr = 0.0f;
    float outputTrimLin = dbToLinear(config.output_trim_db);
    float saturateDriveLin = dbToLinear(config.saturate_drive_db);

    for (size_t i = 0; i < n; ++i) {
        float scSample = processBiquad(scHp, scHpState, audio[i]);
        scSample = processBiquad(scLp, scLpState, scSample);

        float env = envFollower_.processSample(
            scSample, config.detector_type,
            config.attack_ms, config.release_ms, sampleRate);

        if (lookaheadSamples > 0) {
            envDelayBuf[delayWriteIdx] = env;
            float delayedEnv = envDelayBuf[(delayWriteIdx + 1) % lookaheadSamples];
            env = delayedEnv;

            audioDelayBuf[delayWriteIdx] = audio[i];
            audio[i] = audioDelayBuf[(delayWriteIdx + 1) % lookaheadSamples];

            delayWriteIdx = (delayWriteIdx + 1) % lookaheadSamples;
        }

        float grDb = gainComputer_.computeGainDb(
            env, config.threshold_db, config.ratio, config.knee_db);

        float smoothGrDb = gainComputer_.smoothGainDb(
            grDb, config.attack_ms, config.release_ms,
            config.hold_ms, sampleRate);

        float gainLin = dbToLinear(smoothGrDb);
        float outSample = audio[i] * gainLin;

        if (config.saturate_drive_db > 0.0f) {
            float x = outSample * saturateDriveLin;
            x = std::tanh(x);
            outSample = x / saturateDriveLin;
        }

        outSample = outSample * config.wet_dry_mix + audio[i] * (1.0f - config.wet_dry_mix);

        outSample *= outputTrimLin;

        audio[i] = outSample;

        float absGr = std::fabs(smoothGrDb);
        sumGrAbs += absGr;
        if (absGr > maxGr) maxGr = absGr;
    }

    result.outputPeakDb = peakDb(audio.data(), n);
    result.outputRmsDb = rmsDb(audio.data(), n);
    result.gainReductionDb = maxGr;
    result.avgGainReductionDb = sumGrAbs / static_cast<float>(n);
    result.framesProcessed = n;
    result.success = true;

    return result;
}

CompResult Compressor::process(std::vector<float>& audio, int sampleRate) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = processImpl(audio, sampleRate, cfg_);
    auto t1 = std::chrono::high_resolution_clock::now();
    result.processingTimeMs =
        std::chrono::duration<float, std::milli>(t1 - t0).count();
    return result;
}

CompResult Compressor::process(std::vector<float>& audio, int sampleRate,
                                const CompConfig& config) {
    cfg_ = config;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = processImpl(audio, sampleRate, cfg_);
    auto t1 = std::chrono::high_resolution_clock::now();
    result.processingTimeMs =
        std::chrono::duration<float, std::milli>(t1 - t0).count();
    return result;
}

CompResult Compressor::processFile(const std::string& inputPath,
                                    const std::string& outputPath) {
    CompResult result;

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
        fprintf(stderr, "[compress] Input: %s\n", inputPath.c_str());
        fprintf(stderr, "[compress]   Channels:    %d\n", channels);
        fprintf(stderr, "[compress]   Sample rate: %d Hz\n", sampleRate);
        fprintf(stderr, "[compress]   Frames:      %lld\n",
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
            fprintf(stderr, "[compress] Stereo link enabled (%.2f)\n", cfg_.stereo_link);
        }

        envFollower_.reset();
        gainComputer_.reset();

        BiquadCoeffs scHp = designHp(cfg_.sidechain_hp_hz, sampleRate);
        BiquadCoeffs scLp = designLp(cfg_.sidechain_lp_hz, sampleRate);
        BiquadState scHpStateL, scLpStateL, scHpStateR, scLpStateR;

        int lookaheadSamples = static_cast<int>(
            cfg_.lookahead_ms * static_cast<float>(sampleRate) * 0.001f);
        std::vector<float> envDelayBuf(lookaheadSamples, 0.0f);
        std::vector<float> audioDelayBufL(lookaheadSamples, 0.0f);
        std::vector<float> audioDelayBufR(lookaheadSamples, 0.0f);
        int delayWriteIdx = 0;

        float sumGrAbs = 0.0f;
        float maxGr = 0.0f;
        float outputTrimLin = dbToLinear(cfg_.output_trim_db);
        float saturateDriveLin = dbToLinear(cfg_.saturate_drive_db);

        auto t0 = std::chrono::high_resolution_clock::now();

        for (sf_count_t f = 0; f < totalFrames; ++f) {
            float scL = processBiquad(scHp, scHpStateL, channelBufs[0][f]);
            scL = processBiquad(scLp, scLpStateL, scL);
            float envL = envFollower_.processSample(
                scL, cfg_.detector_type,
                cfg_.attack_ms, cfg_.release_ms, sampleRate);

            float scR = processBiquad(scHp, scHpStateR, channelBufs[1][f]);
            scR = processBiquad(scLp, scLpStateR, scR);
            float envR = envFollower_.processSample(
                scR, cfg_.detector_type,
                cfg_.attack_ms, cfg_.release_ms, sampleRate);

            float env = envL * cfg_.stereo_link + envR * (1.0f - cfg_.stereo_link);

            if (lookaheadSamples > 0) {
                envDelayBuf[delayWriteIdx] = env;
                float delayedEnv = envDelayBuf[(delayWriteIdx + 1) % lookaheadSamples];
                env = delayedEnv;

                audioDelayBufL[delayWriteIdx] = channelBufs[0][f];
                audioDelayBufR[delayWriteIdx] = channelBufs[1][f];
                channelBufs[0][f] = audioDelayBufL[(delayWriteIdx + 1) % lookaheadSamples];
                channelBufs[1][f] = audioDelayBufR[(delayWriteIdx + 1) % lookaheadSamples];

                delayWriteIdx = (delayWriteIdx + 1) % lookaheadSamples;
            }

            float grDb = gainComputer_.computeGainDb(
                env, cfg_.threshold_db, cfg_.ratio, cfg_.knee_db);

            float smoothGrDb = gainComputer_.smoothGainDb(
                grDb, cfg_.attack_ms, cfg_.release_ms,
                cfg_.hold_ms, sampleRate);

            float gainLin = dbToLinear(smoothGrDb);

            for (int ch = 0; ch < 2; ++ch) {
                float outSample = channelBufs[ch][f] * gainLin;

                if (cfg_.saturate_drive_db > 0.0f) {
                    float x = outSample * saturateDriveLin;
                    x = std::tanh(x);
                    outSample = x / saturateDriveLin;
                }

                outSample = outSample * cfg_.wet_dry_mix
                          + channelBufs[ch][f] * (1.0f - cfg_.wet_dry_mix);
                outSample *= outputTrimLin;

                channelBufs[ch][f] = outSample;
            }

            float absGr = std::fabs(smoothGrDb);
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

        result.gainReductionDb = maxGr;
        result.avgGainReductionDb = sumGrAbs / static_cast<float>(totalFrames);
        result.framesProcessed = static_cast<size_t>(totalFrames) * 2;
        result.success = true;
    } else {
        for (int ch = 0; ch < channels; ++ch) {
            if (cfg_.verbose) {
                fprintf(stderr, "[compress] Processing channel %d/%d...\n",
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
        fprintf(stderr, "[compress] Output written: %s\n", outputPath.c_str());
    }

    return result;
}

}
