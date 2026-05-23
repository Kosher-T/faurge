#include "faurge/biquad.hpp"

#include <algorithm>
#include <cmath>

namespace faurge {

static constexpr float PI = 3.14159265358979f;

Biquad::Biquad() {
    reset();
}

void Biquad::reset() {
    state_ = {};
}

void Biquad::setFilter(float freq, float q, int sampleRate) {
    coeffs_ = designBandpass(freq, q, sampleRate);
}

void Biquad::process(std::vector<float>& audio) {
    processRaw(audio.data(), audio.data(), audio.size());
}

float Biquad::processSample(float input) {
    float y = coeffs_.b0 * input
            + coeffs_.b1 * state_.x1
            + coeffs_.b2 * state_.x2
            - coeffs_.a1 * state_.y1
            - coeffs_.a2 * state_.y2;
    state_.x2 = state_.x1;
    state_.x1 = input;
    state_.y2 = state_.y1;
    state_.y1 = y;
    return y;
}

void Biquad::processRaw(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float y = coeffs_.b0 * in[i]
                + coeffs_.b1 * state_.x1
                + coeffs_.b2 * state_.x2
                - coeffs_.a1 * state_.y1
                - coeffs_.a2 * state_.y2;
        state_.x2 = state_.x1;
        state_.x1 = in[i];
        state_.y2 = state_.y1;
        state_.y1 = y;
        out[i] = y;
    }
}

BiquadCoeffs Biquad::designBandpass(float freq, float q, int sampleRate) {
    BiquadCoeffs c;
    float w0 = 2.0f * PI * freq / static_cast<float>(sampleRate);
    float alpha = std::sin(w0) / (2.0f * q);

    c.b0 = alpha;
    c.b1 = 0.0f;
    c.b2 = -alpha;
    float a0 = 1.0f + alpha;
    c.a1 = -2.0f * std::cos(w0);
    c.a2 = 1.0f - alpha;

    c.b0 /= a0; c.b1 /= a0; c.b2 /= a0;
    c.a1 /= a0; c.a2 /= a0;
    return c;
}

} // namespace faurge
