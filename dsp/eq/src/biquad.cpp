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

void Biquad::setFilter(FilterType type, float freq, float gainDb,
                       float q, int sampleRate) {
    switch (type) {
        case FilterType::peak:
            coeffs_ = designPeak(freq, gainDb, q, sampleRate);
            break;
        case FilterType::low_shelf:
            coeffs_ = designLowShelf(freq, gainDb, q, sampleRate);
            break;
        case FilterType::high_shelf:
            coeffs_ = designHighShelf(freq, gainDb, q, sampleRate);
            break;
        case FilterType::highpass:
            coeffs_ = designHighpass(freq, q, sampleRate);
            break;
        case FilterType::lowpass:
            coeffs_ = designLowpass(freq, q, sampleRate);
            break;
        case FilterType::bandpass:
            coeffs_ = designBandpass(freq, q, sampleRate);
            break;
        case FilterType::notch:
            coeffs_ = designNotch(freq, q, sampleRate);
            break;
    }
}

void Biquad::process(std::vector<float>& audio) {
    processRaw(audio.data(), audio.data(), audio.size());
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

BiquadCoeffs Biquad::designPeak(float freq, float gainDb,
                                float q, int sampleRate) {
    BiquadCoeffs c;
    float A = std::pow(10.0f, gainDb / 40.0f);
    float w0 = 2.0f * PI * freq / static_cast<float>(sampleRate);
    float alpha = std::sin(w0) / (2.0f * q);

    c.b0 = 1.0f + alpha * A;
    c.b1 = -2.0f * std::cos(w0);
    c.b2 = 1.0f - alpha * A;
    float a0 = 1.0f + alpha / A;
    c.a1 = -2.0f * std::cos(w0);
    c.a2 = 1.0f - alpha / A;

    c.b0 /= a0; c.b1 /= a0; c.b2 /= a0;
    c.a1 /= a0; c.a2 /= a0;
    return c;
}

BiquadCoeffs Biquad::designLowShelf(float freq, float gainDb,
                                    float q, int sampleRate) {
    BiquadCoeffs c;
    float A = std::pow(10.0f, gainDb / 40.0f);
    float w0 = 2.0f * PI * freq / static_cast<float>(sampleRate);
    float alpha = std::sin(w0) / (2.0f * q);
    float cosW0 = std::cos(w0);
    float sqrtA = std::sqrt(A);

    c.b0 = A * ((A + 1.0f) - (A - 1.0f) * cosW0 + 2.0f * sqrtA * alpha);
    c.b1 = 2.0f * A * ((A - 1.0f) - (A + 1.0f) * cosW0);
    c.b2 = A * ((A + 1.0f) - (A - 1.0f) * cosW0 - 2.0f * sqrtA * alpha);
    float a0 = (A + 1.0f) + (A - 1.0f) * cosW0 + 2.0f * sqrtA * alpha;
    c.a1 = -2.0f * ((A - 1.0f) + (A + 1.0f) * cosW0);
    c.a2 = (A + 1.0f) + (A - 1.0f) * cosW0 - 2.0f * sqrtA * alpha;

    c.b0 /= a0; c.b1 /= a0; c.b2 /= a0;
    c.a1 /= a0; c.a2 /= a0;
    return c;
}

BiquadCoeffs Biquad::designHighShelf(float freq, float gainDb,
                                     float q, int sampleRate) {
    BiquadCoeffs c;
    float A = std::pow(10.0f, gainDb / 40.0f);
    float w0 = 2.0f * PI * freq / static_cast<float>(sampleRate);
    float alpha = std::sin(w0) / (2.0f * q);
    float cosW0 = std::cos(w0);
    float sqrtA = std::sqrt(A);

    c.b0 = A * ((A + 1.0f) + (A - 1.0f) * cosW0 + 2.0f * sqrtA * alpha);
    c.b1 = -2.0f * A * ((A - 1.0f) + (A + 1.0f) * cosW0);
    c.b2 = A * ((A + 1.0f) + (A - 1.0f) * cosW0 - 2.0f * sqrtA * alpha);
    float a0 = (A + 1.0f) - (A - 1.0f) * cosW0 + 2.0f * sqrtA * alpha;
    c.a1 = 2.0f * ((A - 1.0f) - (A + 1.0f) * cosW0);
    c.a2 = (A + 1.0f) - (A - 1.0f) * cosW0 - 2.0f * sqrtA * alpha;

    c.b0 /= a0; c.b1 /= a0; c.b2 /= a0;
    c.a1 /= a0; c.a2 /= a0;
    return c;
}

BiquadCoeffs Biquad::designHighpass(float freq, float q, int sampleRate) {
    BiquadCoeffs c;
    float w0 = 2.0f * PI * freq / static_cast<float>(sampleRate);
    float alpha = std::sin(w0) / (2.0f * q);
    float cosW0 = std::cos(w0);

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

BiquadCoeffs Biquad::designLowpass(float freq, float q, int sampleRate) {
    BiquadCoeffs c;
    float w0 = 2.0f * PI * freq / static_cast<float>(sampleRate);
    float alpha = std::sin(w0) / (2.0f * q);
    float cosW0 = std::cos(w0);

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

BiquadCoeffs Biquad::designNotch(float freq, float q, int sampleRate) {
    BiquadCoeffs c;
    float w0 = 2.0f * PI * freq / static_cast<float>(sampleRate);
    float alpha = std::sin(w0) / (2.0f * q);
    float cosW0 = std::cos(w0);

    c.b0 = 1.0f;
    c.b1 = -2.0f * cosW0;
    c.b2 = 1.0f;
    float a0 = 1.0f + alpha;
    c.a1 = -2.0f * cosW0;
    c.a2 = 1.0f - alpha;

    c.b0 /= a0; c.b1 /= a0; c.b2 /= a0;
    c.a1 /= a0; c.a2 /= a0;
    return c;
}

}
