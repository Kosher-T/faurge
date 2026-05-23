#pragma once

#include <cstddef>

namespace faurge {

struct BiquadCoeffs {
    float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f;
    float a1 = 0.0f, a2 = 0.0f;
};

struct BiquadState {
    float x1 = 0.0f, x2 = 0.0f, y1 = 0.0f, y2 = 0.0f;
};

BiquadCoeffs designLp(float freq, float q, int sampleRate);
BiquadCoeffs designButterworthLpStage(float cutoff, int sampleRate, float q);
float processBiquad(const BiquadCoeffs& c, BiquadState& s, float in);
void resetBiquad(BiquadState& s);

} // namespace faurge
