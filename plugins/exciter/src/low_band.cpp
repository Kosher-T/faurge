#include "faurge/low_band.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace faurge {

static constexpr float PI = 3.14159265358979f;

static constexpr float SUB_LPF_HZ = 120.0f;

LowBand::LowBand() {}

void LowBand::initLpf(int sampleRate) {
    prevSampleRate_ = sampleRate;
    float w0 = 2.0f * PI * SUB_LPF_HZ / static_cast<float>(sampleRate);
    float alpha = std::sin(w0) / 1.41421356f;
    prevLpCoeff_ = w0;

    lp1_ = {}; lp2_ = {};
}

void LowBand::applyLpf(const float* in, float* out, size_t n) {
    float w0 = 2.0f * PI * SUB_LPF_HZ / static_cast<float>(prevSampleRate_);
    float alpha = std::sin(w0) / 1.41421356f;

    float b0 = (1.0f - std::cos(w0)) / 2.0f;
    float b1 = 1.0f - std::cos(w0);
    float b2 = b0;
    float a0 = 1.0f + alpha;
    float a1 = -2.0f * std::cos(w0);
    float a2 = 1.0f - alpha;

    b0 /= a0; b1 /= a0; b2 /= a0;
    a1 /= a0; a2 /= a0;

    std::vector<float> tmp(n);

    for (size_t i = 0; i < n; ++i) {
        float y = b0 * in[i] + b1 * lp1_.x1 + b2 * lp1_.x2
                - a1 * lp1_.y1 - a2 * lp1_.y2;
        lp1_.x2 = lp1_.x1; lp1_.x1 = in[i];
        lp1_.y2 = lp1_.y1; lp1_.y1 = y;
        tmp[i] = y;
    }

    for (size_t i = 0; i < n; ++i) {
        float y = b0 * tmp[i] + b1 * lp2_.x1 + b2 * lp2_.x2
                - a1 * lp2_.y1 - a2 * lp2_.y2;
        lp2_.x2 = lp2_.x1; lp2_.x1 = tmp[i];
        lp2_.y2 = lp2_.y1; lp2_.y1 = y;
        out[i] = y;
    }
}

void LowBand::process(const float* input, float* output,
                      size_t numSamples, int sampleRate,
                      float driveDb, float subLevel) {
    if (numSamples == 0) return;

    float driveLinear = std::pow(10.0f, driveDb / 20.0f);

    if (subLevel <= 0.0f && driveDb < 0.01f) {
        std::memcpy(output, input, numSamples * sizeof(float));
        return;
    }

    if (sampleRate != prevSampleRate_) {
        initLpf(sampleRate);
    }

    if (rectBuf_.size() < numSamples) {
        rectBuf_.resize(numSamples);
        lpBuf_.resize(numSamples);
    }

    for (size_t i = 0; i < numSamples; ++i) {
        rectBuf_[i] = std::fabs(input[i]) * driveLinear;
    }

    applyLpf(rectBuf_.data(), lpBuf_.data(), numSamples);

    for (size_t i = 0; i < numSamples; ++i) {
        output[i] = lpBuf_[i] * subLevel;
    }
}

}
