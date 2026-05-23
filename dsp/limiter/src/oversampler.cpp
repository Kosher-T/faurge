#include "faurge/oversampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace faurge {

namespace {

constexpr float BUTTER_Q1 = 0.5412f;
constexpr float BUTTER_Q2 = 1.3066f;

void apply4thOrderLp(BiquadCoeffs& c1, BiquadCoeffs& c2,
                     BiquadState& s1, BiquadState& s2,
                     float* buf, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float tmp = processBiquad(c1, s1, buf[i]);
        buf[i] = processBiquad(c2, s2, tmp);
    }
}

} // anonymous namespace

Oversampler::Oversampler() {}

void Oversampler::configureFilters(int inputSampleRate, int factor) {
    if (factor <= 1) return;

    int upRate = inputSampleRate * factor;
    float cutoff = 0.45f * static_cast<float>(inputSampleRate) / 2.0f;

    upCoeffs_ = designButterworthLpStage(cutoff, upRate, BUTTER_Q1);
    BiquadCoeffs upC2 = designButterworthLpStage(cutoff, upRate, BUTTER_Q2);

    downCoeffs_ = designButterworthLpStage(cutoff, upRate, BUTTER_Q1);
    BiquadCoeffs downC2 = designButterworthLpStage(cutoff, upRate, BUTTER_Q2);

    upState1_ = {}; upState2_ = {};
    downState1_ = {}; downState2_ = {};
    currentFactor_ = factor;
}

LimiterResult Oversampler::process(std::vector<float>& audio, int sampleRate,
                                   int oversamplingFactor, ProcessFn fn) {
    if (oversamplingFactor <= 1 || audio.empty()) {
        return fn(audio, sampleRate);
    }

    int factor = oversamplingFactor;
    if (factor > 4) factor = 4;
    if (factor < 2) factor = 1;

    configureFilters(sampleRate, factor);

    size_t n = audio.size();
    size_t upN = n * static_cast<size_t>(factor);
    int upRate = sampleRate * factor;

    std::vector<float> upBuf(upN, 0.0f);

    if (factor == 2) {
        for (size_t i = 0; i < n; ++i) {
            upBuf[i * 2] = audio[i];
        }
    } else if (factor == 4) {
        for (size_t i = 0; i < n; ++i) {
            upBuf[i * 4] = audio[i];
        }
    }

    BiquadCoeffs upC2 = designButterworthLpStage(
        0.45f * static_cast<float>(sampleRate) / 2.0f, upRate, 1.3066f);
    BiquadState upS1 = {}, upS2 = {}, upS3 = {}, upS4 = {};
    BiquadState downS1 = {}, downS2 = {}, downS3 = {}, downS4 = {};

    if (factor == 2) {
        {
            BiquadState s1 = {}, s2 = {};
            for (size_t i = 0; i < upN; ++i) {
                float tmp = processBiquad(upCoeffs_, s1, upBuf[i]);
                upBuf[i] = processBiquad(upC2, s2, tmp);
            }
        }

        auto result = fn(upBuf, upRate);

        {
            BiquadState s1 = {}, s2 = {};
            for (size_t i = 0; i < upN; ++i) {
                float tmp = processBiquad(downCoeffs_, s1, upBuf[i]);
                upBuf[i] = processBiquad(upC2, s2, tmp);
            }
        }

        for (size_t i = 0; i < n; ++i) {
            audio[i] = upBuf[i * 2];
        }

        result.framesProcessed = n;
        return result;
    }

    if (factor == 4) {
        {
            BiquadState s1 = {}, s2 = {};
            for (size_t i = 0; i < upN; ++i) {
                float tmp = processBiquad(upCoeffs_, s1, upBuf[i]);
                upBuf[i] = processBiquad(upC2, s2, tmp);
            }
        }

        int upRate2 = sampleRate * 2;
        size_t midN = n * 2;
        std::vector<float> midBuf(midN * 2, 0.0f);
        for (size_t i = 0; i < midN; ++i) {
            midBuf[i * 2] = upBuf[i];
        }

        BiquadCoeffs midC1 = designButterworthLpStage(
            0.45f * static_cast<float>(sampleRate) / 2.0f, upRate2 * 2, 0.5412f);
        BiquadCoeffs midC2 = designButterworthLpStage(
            0.45f * static_cast<float>(sampleRate) / 2.0f, upRate2 * 2, 1.3066f);

        {
            BiquadState s1 = {}, s2 = {};
            for (size_t i = 0; i < upN; ++i) {
                float tmp = processBiquad(midC1, s1, upBuf[i]);
                upBuf[i] = processBiquad(midC2, s2, tmp);
            }
        }

        auto result = fn(upBuf, upRate);

        {
            BiquadState s1 = {}, s2 = {};
            for (size_t i = 0; i < upN; ++i) {
                float tmp = processBiquad(midC1, s1, upBuf[i]);
                upBuf[i] = processBiquad(midC2, s2, tmp);
            }
        }

        for (size_t i = 0; i < midN; ++i) {
            midBuf[i] = upBuf[i * 2];
        }

        {
            BiquadState s1 = {}, s2 = {};
            for (size_t i = 0; i < midN; ++i) {
                float tmp = processBiquad(downCoeffs_, s1, midBuf[i]);
                midBuf[i] = processBiquad(upC2, s2, tmp);
            }
        }

        for (size_t i = 0; i < n; ++i) {
            audio[i] = midBuf[i * 2];
        }

        result.framesProcessed = n;
        return result;
    }

    return fn(audio, sampleRate);
}

} // namespace faurge
