#include "faurge/saturator.hpp"
#include "faurge/waveshaper.hpp"
#include "faurge/sat_metrics.hpp"

#include <sndfile.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace faurge {

namespace {

constexpr float BUTTER_Q1 = 0.5412f;
constexpr float BUTTER_Q2 = 1.3066f;

float clamp(float val, float lo, float hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

} // anonymous namespace

Saturator::Saturator(const SatConfig& config) : cfg_(config) {}

float Saturator::dbToLinear(float db) {
    return std::pow(10.0f, db / 20.0f);
}

float Saturator::linearToDb(float linear) {
    if (linear < 1e-30f) return -120.0f;
    return 20.0f * std::log10(linear);
}

float Saturator::peakDb(const float* audio, size_t n) {
    float peak = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float absVal = std::fabs(audio[i]);
        if (absVal > peak) peak = absVal;
    }
    if (peak < 1e-30f) return -120.0f;
    return 20.0f * std::log10(peak);
}

float Saturator::rmsDb(const float* audio, size_t n) {
    if (n == 0) return -120.0f;
    float sumSq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sumSq += audio[i] * audio[i];
    }
    float rms = std::sqrt(sumSq / static_cast<float>(n));
    if (rms < 1e-30f) return -120.0f;
    return 20.0f * std::log10(rms);
}

void Saturator::applyWaveshaper(float* buf, size_t n, float& harmSum) {
    auto fn = getWaveshaper(cfg_.sat_type);
    harmSum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float in = buf[i];
        float out = fn(in);
        harmSum += std::fabs(out - in);
        buf[i] = out;
    }
}

void Saturator::applyBiquads(float* buf, size_t n,
                              const BiquadCoeffs& c1,
                              const BiquadCoeffs& c2) {
    BiquadState s1 = {}, s2 = {};
    for (size_t i = 0; i < n; ++i) {
        float tmp = processBiquad(c1, s1, buf[i]);
        buf[i] = processBiquad(c2, s2, tmp);
    }
}

void Saturator::processOversampled(float* buf, size_t n, int sampleRate) {
    int factor = cfg_.oversampling;
    if (factor > 4) factor = 4;
    if (factor < 2) {
        float harmDummy = 0.0f;
        applyWaveshaper(buf, n, harmDummy);
        return;
    }

    float aaCutoff = 0.45f * static_cast<float>(sampleRate) / 2.0f;
    BiquadCoeffs aaC1 = designButterworthLpStage(aaCutoff, sampleRate * 2, BUTTER_Q1);
    BiquadCoeffs aaC2 = designButterworthLpStage(aaCutoff, sampleRate * 2, BUTTER_Q2);

    if (factor == 2) {
        size_t upN = n * 2;
        std::vector<float> upBuf(upN, 0.0f);
        for (size_t i = 0; i < n; ++i) upBuf[i * 2] = buf[i];

        applyBiquads(upBuf.data(), upN, aaC1, aaC2);

        float harmDummy = 0.0f;
        applyWaveshaper(upBuf.data(), upN, harmDummy);

        applyBiquads(upBuf.data(), upN, aaC1, aaC2);

        for (size_t i = 0; i < n; ++i) buf[i] = upBuf[i * 2];
        return;
    }

    float harmDummy = 0.0f;
    applyWaveshaper(buf, n, harmDummy);
}

void Saturator::processWetPath(float* buf, size_t n, int sampleRate,
                                float& harmSum) {
    BiquadState hpfState = {}, lpfState = {};

    float hpfHz = clamp(cfg_.hpf_hz, 20.0f, cfg_.lpf_hz - 1.0f);
    float lpfHz = clamp(cfg_.lpf_hz, hpfHz + 1.0f, 20000.0f);

    BiquadCoeffs hpfC = designHp(hpfHz, 0.7071f, sampleRate);
    BiquadCoeffs lpfC = designLp(lpfHz, 0.7071f, sampleRate);

    for (size_t i = 0; i < n; ++i) {
        buf[i] = processBiquad(hpfC, hpfState, buf[i]);
    }

    float driveLin = dbToLinear(cfg_.drive_db);
    for (size_t i = 0; i < n; ++i) {
        buf[i] *= driveLin;
    }

    if (cfg_.oversampling > 1) {
        float oversampledHarmSum = 0.0f;
        processOversampled(buf, n, sampleRate);
        harmSum += oversampledHarmSum;
    } else {
        applyWaveshaper(buf, n, harmSum);
    }

    for (size_t i = 0; i < n; ++i) {
        buf[i] = processBiquad(lpfC, lpfState, buf[i]);
    }
}

SatResult Saturator::processImpl(std::vector<float>& audio, int sampleRate,
                                  const SatConfig& config) {
    SatResult result;

    if (audio.empty() || sampleRate <= 0) {
        result.success = false;
        result.errorMessage = "Empty audio buffer or invalid sample rate";
        return result;
    }

    size_t n = audio.size();

    result.inputPeakDb = peakDb(audio.data(), n);
    result.inputRmsDb = rmsDb(audio.data(), n);

    std::vector<float> wet = audio;

    float totalHarmSum = 0.0f;
    processWetPath(wet.data(), n, sampleRate, totalHarmSum);

    float mixVal = clamp(config.mix, 0.0f, 1.0f);
    float trimLin = dbToLinear(config.output_trim_db);

    float dcSum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        audio[i] = (mixVal * wet[i] + (1.0f - mixVal) * audio[i]) * trimLin;
        dcSum += audio[i];
    }

    result.outputPeakDb = peakDb(audio.data(), n);
    result.outputRmsDb = rmsDb(audio.data(), n);
    result.avgHarmonicDb = linearToDb(totalHarmSum / static_cast<float>(n));
    result.dcOffset = dcSum / static_cast<float>(n);
    result.framesProcessed = n;
    result.success = true;

    return result;
}

SatResult Saturator::process(std::vector<float>& audio, int sampleRate) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = processImpl(audio, sampleRate, cfg_);
    auto t1 = std::chrono::high_resolution_clock::now();
    result.processingTimeMs =
        std::chrono::duration<float, std::milli>(t1 - t0).count();
    return result;
}

SatResult Saturator::process(std::vector<float>& audio, int sampleRate,
                              const SatConfig& config) {
    cfg_ = config;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = processImpl(audio, sampleRate, cfg_);
    auto t1 = std::chrono::high_resolution_clock::now();
    result.processingTimeMs =
        std::chrono::duration<float, std::milli>(t1 - t0).count();
    return result;
}

SatResult Saturator::processFile(const std::string& inputPath,
                                  const std::string& outputPath) {
    SatResult result;

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
        fprintf(stderr, "[saturate] Input: %s\n", inputPath.c_str());
        fprintf(stderr, "[saturate]   Channels:    %d\n", channels);
        fprintf(stderr, "[saturate]   Sample rate: %d Hz\n", sampleRate);
        fprintf(stderr, "[saturate]   Frames:      %lld\n",
                static_cast<long long>(totalFrames));
        fprintf(stderr, "[saturate]   Type:        %s\n",
                satTypeToString(static_cast<SatType>(cfg_.sat_type)));
        fprintf(stderr, "[saturate]   Drive:       %.1f dB\n", cfg_.drive_db);
        fprintf(stderr, "[saturate]   HPF:         %.0f Hz\n", cfg_.hpf_hz);
        fprintf(stderr, "[saturate]   LPF:         %.0f Hz\n", cfg_.lpf_hz);
        fprintf(stderr, "[saturate]   Mix:         %.2f\n", cfg_.mix);
        fprintf(stderr, "[saturate]   Oversampling:%d x\n", cfg_.oversampling);
        fprintf(stderr, "[saturate]   Output trim: %.1f dB\n", cfg_.output_trim_db);
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

    auto t0 = std::chrono::high_resolution_clock::now();

    float totalHarmSum = 0.0f;
    double dcSum = 0.0;

    for (int ch = 0; ch < channels; ++ch) {
        if (cfg_.verbose) {
            fprintf(stderr, "[saturate] Processing channel %d/%d...\n",
                    ch + 1, channels);
        }

        float chHarmSum = 0.0f;
        processWetPath(channelBufs[ch].data(), static_cast<size_t>(totalFrames),
                       sampleRate, chHarmSum);
        totalHarmSum += chHarmSum;

        if (cfg_.verbose) {
            float chPeak = peakDb(channelBufs[ch].data(),
                                  static_cast<size_t>(totalFrames));
            fprintf(stderr, "[saturate]   Channel %d peak: %.1f dB\n",
                    ch + 1, chPeak);
        }
    }

    float mixVal = clamp(cfg_.mix, 0.0f, 1.0f);
    float trimLin = dbToLinear(cfg_.output_trim_db);

    for (int ch = 0; ch < channels; ++ch) {
        for (sf_count_t f = 0; f < totalFrames; ++f) {
            float dry = interleavedBuf[f * channels + ch];
            float wetVal = channelBufs[ch][f];
            float out = (mixVal * wetVal + (1.0f - mixVal) * dry) * trimLin;
            channelBufs[ch][f] = out;
            interleavedBuf[f * channels + ch] = out;
            dcSum += static_cast<double>(out);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.processingTimeMs =
        std::chrono::duration<float, std::milli>(t1 - t0).count();

    result.outputPeakDb = peakDb(interleavedBuf.data(), totalSamples);
    result.outputRmsDb = rmsDb(interleavedBuf.data(), totalSamples);
    result.avgHarmonicDb = linearToDb(totalHarmSum / static_cast<float>(totalSamples));
    result.dcOffset = static_cast<float>(dcSum / static_cast<double>(totalSamples));
    result.framesProcessed = totalSamples;
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
        fprintf(stderr, "[saturate] Output written: %s\n", outputPath.c_str());
    }

    return result;
}

} // namespace faurge
