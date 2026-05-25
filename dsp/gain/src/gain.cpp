#include "faurge/gain.hpp"
#include "faurge/gain_metrics.hpp"

#include <sndfile.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace faurge {

Gain::Gain(const GainConfig& config) : cfg_(config) {}

float Gain::dbToLinear(float db) {
    return std::pow(10.0f, db / 20.0f);
}

float Gain::linearToDb(float linear) {
    if (linear < 1e-30f) return -120.0f;
    return 20.0f * std::log10(linear);
}

float Gain::peakDb(const float* audio, size_t n) {
    float peak = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float absVal = std::fabs(audio[i]);
        if (absVal > peak) peak = absVal;
    }
    if (peak < 1e-30f) return -120.0f;
    return 20.0f * std::log10(peak);
}

float Gain::rmsDb(const float* audio, size_t n) {
    if (n == 0) return -120.0f;
    double sumSq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sumSq += static_cast<double>(audio[i]) * audio[i];
    }
    double rms = std::sqrt(sumSq / static_cast<double>(n));
    if (rms < 1e-30) return -120.0f;
    return 20.0f * std::log10(static_cast<float>(rms));
}

// ─── K-weighting filter design (ITU-R BS.1770-4) ───

// Stage 1: Pre-filter, 1st-order HPF
// H(s) = s / (s + 129.4)
// Bilinear transform without pre-warping:
//   b0 = A/(A+ω), b1 = -A/(A+ω), a1 = (-A+ω)/(A+ω)
//   where A = 2*sampleRate, ω = 129.4
static void designPreFilter(int sampleRate, float& b0, float& b1, float& a1) {
    float fs = static_cast<float>(sampleRate);
    float omega_a = 129.4f;
    float A = 2.0f * fs;
    float denom = A + omega_a;
    b0 = A / denom;
    b1 = -A / denom;
    a1 = (-A + omega_a) / denom;
}

// Stage 2: RLB filter, 2nd-order shelf
// H(s) = (s² + 129.4²) / (s² + 2·18.6·s + 18.6²)
// Bilinear transform:
//   b0 = (A²+ωb²)/D, b1 = 2(ωb²-A²)/D, b2 = (A²+ωb²)/D
//   a1 = 2(ωc²-A²)/D, a2 = (A-ωc)²/D
//   where D = (A+ωc)², ωb = 129.4, ωc = 18.6
static void designRlbFilter(int sampleRate,
                            float& b0, float& b1, float& b2,
                            float& a1, float& a2)
{
    float fs = static_cast<float>(sampleRate);
    float omega_b = 129.4f;
    float omega_c = 18.6f;
    float A = 2.0f * fs;
    float A2 = A * A;
    float wb2 = omega_b * omega_b;
    float wc2 = omega_c * omega_c;
    float d0 = (A + omega_c) * (A + omega_c);
    b0 = (A2 + wb2) / d0;
    b1 = 2.0f * (wb2 - A2) / d0;
    b2 = (A2 + wb2) / d0;
    a1 = 2.0f * (wc2 - A2) / d0;
    a2 = (A - omega_c) * (A - omega_c) / d0;
}

// ─── BS.1770-4 integrated LUFS measurement ───

float Gain::measureLufs(const float* audio, size_t totalSamples,
                        int sampleRate, int channels)
{
    if (totalSamples == 0 || sampleRate <= 0 || channels <= 0) return -120.0f;

    size_t frames = totalSamples / channels;
    size_t blockSize = static_cast<size_t>(0.4f * sampleRate);
    if (blockSize < 1) blockSize = 1;

    float pre_b0, pre_b1, pre_a1;
    designPreFilter(sampleRate, pre_b0, pre_b1, pre_a1);

    float rlb_b0, rlb_b1, rlb_b2, rlb_a1, rlb_a2;
    designRlbFilter(sampleRate, rlb_b0, rlb_b1, rlb_b2, rlb_a1, rlb_a2);

    size_t numBlocks = frames / blockSize;
    if (numBlocks == 0) {
        numBlocks = 1;
        blockSize = frames;
    }

    std::vector<double> blockMeans(numBlocks, 0.0);

    for (int ch = 0; ch < channels; ++ch) {
        // Per-channel K-weighting filter state (continuous across blocks)
        float pre_z1 = 0.0f;
        float rlb_z1 = 0.0f;
        float rlb_z2 = 0.0f;

        for (size_t block = 0; block < numBlocks; ++block) {
            size_t blockStart = block * blockSize;
            size_t blockFrames = std::min(blockSize, frames - blockStart);

            double sumSq = 0.0;

            for (size_t i = 0; i < blockFrames; ++i) {
                size_t idx = (blockStart + i) * static_cast<size_t>(channels) + static_cast<size_t>(ch);
                float x = audio[idx];

                // Stage 1: Pre-filter (DF2T, 1st-order)
                float y_pre = pre_b0 * x + pre_z1;
                pre_z1 = pre_b1 * x - pre_a1 * y_pre;

                // Stage 2: RLB filter (DF2T, 2nd-order)
                float y_rlb = rlb_b0 * y_pre + rlb_z1;
                rlb_z1 = rlb_b1 * y_pre - rlb_a1 * y_rlb + rlb_z2;
                rlb_z2 = rlb_b2 * y_pre - rlb_a2 * y_rlb;

                sumSq += static_cast<double>(y_rlb) * y_rlb;
            }

            double meanSq = sumSq / static_cast<double>(blockFrames);
            double weight = (ch < 2) ? 1.0 : 0.0;
            blockMeans[block] += meanSq * weight;
        }
    }

    // ─── Gating pass 1: absolute gate (-70 LUFS) ───
    double absThreshold = std::pow(10.0, (-70.0 + 0.691) / 10.0);

    std::vector<double> gatedMeans;
    gatedMeans.reserve(numBlocks);
    for (size_t b = 0; b < numBlocks; ++b) {
        if (blockMeans[b] > absThreshold) {
            gatedMeans.push_back(blockMeans[b]);
        }
    }

    if (gatedMeans.empty()) return -120.0f;

    double sumGated = 0.0;
    for (size_t i = 0; i < gatedMeans.size(); ++i) sumGated += gatedMeans[i];
    double meanGated = sumGated / static_cast<double>(gatedMeans.size());
    double lufsGated = -0.691 + 10.0 * std::log10(meanGated);

    // ─── Gating pass 2: relative gate (gated - 10 dB) ───
    // Only apply if we have more than one block
    if (gatedMeans.size() <= 1) {
        return static_cast<float>(lufsGated);
    }

    double relThreshold = std::pow(10.0, (lufsGated - 10.0 + 0.691) / 10.0);

    std::vector<double> finalMeans;
    finalMeans.reserve(gatedMeans.size());
    for (size_t i = 0; i < gatedMeans.size(); ++i) {
        if (gatedMeans[i] > relThreshold) {
            finalMeans.push_back(gatedMeans[i]);
        }
    }

    if (finalMeans.empty()) {
        return static_cast<float>(lufsGated);
    }

    double sumFinal = 0.0;
    for (size_t i = 0; i < finalMeans.size(); ++i) sumFinal += finalMeans[i];
    double meanFinal = sumFinal / static_cast<double>(finalMeans.size());
    double lufsFinal = -0.691 + 10.0 * std::log10(meanFinal);

    return static_cast<float>(lufsFinal);
}

GainResult Gain::processImpl(std::vector<float>& audio,
                             int sampleRate,
                             int channels,
                             const GainConfig& config)
{
    GainResult result;

    if (audio.empty() || sampleRate <= 0 || channels <= 0) {
        result.success = false;
        result.errorMessage = "Empty audio buffer, invalid sample rate, or invalid channel count";
        return result;
    }

    size_t n = audio.size();
    size_t frames = n / channels;

    result.inputPeakDb = peakDb(audio.data(), n);
    result.inputRmsDb = rmsDb(audio.data(), n);
    result.inputLufs = measureLufs(audio.data(), n, sampleRate, channels);

    float gainLin = dbToLinear(config.gain_db);
    result.appliedBalance = config.stereo_balance;

    if (channels == 2) {
        float leftBalanceGain, rightBalanceGain;
        if (config.stereo_balance <= 0.0f) {
            leftBalanceGain = 1.0f;
            rightBalanceGain = 1.0f + config.stereo_balance;
        } else {
            leftBalanceGain = 1.0f - config.stereo_balance;
            rightBalanceGain = 1.0f;
        }

        float leftGain = gainLin * leftBalanceGain;
        float rightGain = gainLin * rightBalanceGain;

        for (size_t i = 0; i < frames; ++i) {
            size_t idx = i * 2;
            audio[idx]     *= leftGain;
            audio[idx + 1] *= rightGain;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            audio[i] *= gainLin;
        }
    }

    result.outputPeakDb = peakDb(audio.data(), n);
    result.outputRmsDb = rmsDb(audio.data(), n);
    result.outputLufs = measureLufs(audio.data(), n, sampleRate, channels);
    result.peakChangeDb = result.outputPeakDb - result.inputPeakDb;
    result.rmsChangeDb = result.outputRmsDb - result.inputRmsDb;
    result.clipping = (result.outputPeakDb >= 0.0f);
    result.framesProcessed = frames;
    result.success = true;

    return result;
}

GainResult Gain::process(std::vector<float>& audio, int sampleRate, int channels) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = processImpl(audio, sampleRate, channels, cfg_);
    auto t1 = std::chrono::high_resolution_clock::now();
    result.processingTimeMs =
        std::chrono::duration<float, std::milli>(t1 - t0).count();
    return result;
}

GainResult Gain::process(std::vector<float>& audio, int sampleRate, int channels,
                          const GainConfig& config) {
    cfg_ = config;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = processImpl(audio, sampleRate, channels, cfg_);
    auto t1 = std::chrono::high_resolution_clock::now();
    result.processingTimeMs =
        std::chrono::duration<float, std::milli>(t1 - t0).count();
    return result;
}

GainResult Gain::processFile(const std::string& inputPath,
                              const std::string& outputPath) {
    GainResult result;

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
        fprintf(stderr, "[gain] Input: %s\n", inputPath.c_str());
        fprintf(stderr, "[gain]   Channels:    %d\n", channels);
        fprintf(stderr, "[gain]   Sample rate: %d Hz\n", sampleRate);
        fprintf(stderr, "[gain]   Frames:      %lld\n",
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

    result = process(interleavedBuf, sampleRate, channels);

    if (!result.success) {
        return result;
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
        fprintf(stderr, "[gain] Output written: %s\n", outputPath.c_str());
    }

    return result;
}

} // namespace faurge
