#include "faurge/biquad.hpp"

#include <algorithm>
#include <cmath>

namespace faurge {

static constexpr float PI = 3.14159265358979f;

BiquadCoeffs designLp(float freq, float q, int sampleRate) {
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

BiquadCoeffs designButterworthLpStage(float cutoff, int sampleRate, float q) {
    return designLp(cutoff, q, sampleRate);
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

void resetBiquad(BiquadState& s) {
    s = {};
}

} // namespace faurge
