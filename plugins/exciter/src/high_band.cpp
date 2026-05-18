#include "faurge/high_band.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace faurge {

static constexpr float PI = 3.14159265358979f;

HighBand::HighBand() {}

void HighBand::initAntiAlias(int sampleRate) {
    float fc = static_cast<float>(sampleRate) * 0.45f;
    float w0 = 2.0f * PI * fc / static_cast<float>(2 * sampleRate);
    float alpha = std::sin(w0) / 1.41421356f;

    float b0 = (1.0f - std::cos(w0)) / 2.0f;
    float b1 = 1.0f - std::cos(w0);
    float b2 = b0;
    float a0 = 1.0f + alpha;
    float a1 = -2.0f * std::cos(w0);
    float a2 = 1.0f - alpha;

    aaState_.x1 = aaState_.x2 = aaState_.y1 = aaState_.y2 = 0.0f;
}

void HighBand::applyAntiAlias(float* buf, size_t n) {
    float fc = 0.45f;
    float w0 = 2.0f * PI * fc;
    float alpha = std::sin(w0) / 1.41421356f;

    float b0 = (1.0f - std::cos(w0)) / 2.0f;
    float b1 = 1.0f - std::cos(w0);
    float b2 = b0;
    float a0 = 1.0f + alpha;
    float a1 = -2.0f * std::cos(w0);
    float a2 = 1.0f - alpha;

    b0 /= a0; b1 /= a0; b2 /= a0;
    a1 /= a0; a2 /= a0;

    for (size_t i = 0; i < n; ++i) {
        float y = b0 * buf[i] + b1 * aaState_.x1 + b2 * aaState_.x2
                - a1 * aaState_.y1 - a2 * aaState_.y2;
        aaState_.x2 = aaState_.x1;
        aaState_.x1 = buf[i];
        aaState_.y2 = aaState_.y1;
        aaState_.y1 = y;
        buf[i] = y;
    }
}

void HighBand::process(const float* input, float* output,
                       size_t numSamples, int sampleRate, float driveDb) {
    if (numSamples == 0) return;

    float driveLinear = std::pow(10.0f, driveDb / 20.0f);

    if (driveLinear <= 0.0f || driveDb < 0.01f) {
        std::memcpy(output, input, numSamples * sizeof(float));
        return;
    }

    size_t osLen = numSamples * 2;
    if (oversampleBuf_.size() < osLen) {
        oversampleBuf_.resize(osLen);
        filterBuf_.resize(osLen);
    }

    for (size_t i = 0; i < numSamples; ++i) {
        oversampleBuf_[i * 2]     = input[i];
        oversampleBuf_[i * 2 + 1] = input[i];
    }

    for (size_t i = 0; i < osLen; ++i) {
        float s = oversampleBuf_[i] * driveLinear;
        s = std::tanh(s);
        oversampleBuf_[i] = s;
    }

    aaState_ = {};
    initAntiAlias(sampleRate);
    applyAntiAlias(oversampleBuf_.data(), osLen);

    for (size_t i = 0; i < numSamples; ++i) {
        output[i] = oversampleBuf_[i * 2];
    }
}

}
