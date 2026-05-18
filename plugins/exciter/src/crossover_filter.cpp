#include "faurge/crossover_filter.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace faurge {

static constexpr float PI = 3.14159265358979f;

CrossoverFilter::CrossoverFilter() {
    reset();
}

void CrossoverFilter::reset() {
    lp1_ = {}; lp2_ = {};
    hp1_ = {}; hp2_ = {};
}

void CrossoverFilter::biquadLP(const float* in, float* out, size_t n,
                               BiquadState& s, float b0, float b1, float b2,
                               float a1, float a2) {
    for (size_t i = 0; i < n; ++i) {
        float y = b0 * in[i] + b1 * s.x1 + b2 * s.x2 - a1 * s.y1 - a2 * s.y2;
        s.x2 = s.x1;
        s.x1 = in[i];
        s.y2 = s.y1;
        s.y1 = y;
        out[i] = y;
    }
}

void CrossoverFilter::biquadHP(const float* in, float* out, size_t n,
                               BiquadState& s, float b0, float b1, float b2,
                               float a1, float a2) {
    for (size_t i = 0; i < n; ++i) {
        float y = b0 * in[i] + b1 * s.x1 + b2 * s.x2 - a1 * s.y1 - a2 * s.y2;
        s.x2 = s.x1;
        s.x1 = in[i];
        s.y2 = s.y1;
        s.y1 = y;
        out[i] = y;
    }
}

void CrossoverFilter::process(const float* input, float* lowOut, float* highOut,
                              size_t numSamples, int sampleRate,
                              float crossoverHz) {
    if (numSamples == 0) return;

    std::vector<float> lowTmp, highTmp;
    if (!lowOut) { lowTmp.resize(numSamples); lowOut = lowTmp.data(); }
    if (!highOut) { highTmp.resize(numSamples); highOut = highTmp.data(); }

    if (crossoverHz <= 20.0f) {
        std::memcpy(lowOut, input, numSamples * sizeof(float));
        std::memset(highOut, 0, numSamples * sizeof(float));
        return;
    }

    float nyquist = 0.5f * static_cast<float>(sampleRate);
    if (crossoverHz >= nyquist) {
        std::memset(lowOut, 0, numSamples * sizeof(float));
        std::memcpy(highOut, input, numSamples * sizeof(float));
        return;
    }

    float w0 = 2.0f * PI * crossoverHz / static_cast<float>(sampleRate);
    float alpha = std::sin(w0) / 1.41421356f;

    float b0_lp = (1.0f - std::cos(w0)) / 2.0f;
    float b1_lp = 1.0f - std::cos(w0);
    float b2_lp = b0_lp;
    float a0_lp = 1.0f + alpha;
    float a1_lp = -2.0f * std::cos(w0);
    float a2_lp = 1.0f - alpha;

    b0_lp /= a0_lp; b1_lp /= a0_lp; b2_lp /= a0_lp;
    a1_lp /= a0_lp; a2_lp /= a0_lp;

    float b0_hp = (1.0f + std::cos(w0)) / 2.0f;
    float b1_hp = -(1.0f + std::cos(w0));
    float b2_hp = b0_hp;
    float a0_hp = 1.0f + alpha;
    float a1_hp = -2.0f * std::cos(w0);
    float a2_hp = 1.0f - alpha;

    b0_hp /= a0_hp; b1_hp /= a0_hp; b2_hp /= a0_hp;
    a1_hp /= a0_hp; a2_hp /= a0_hp;

    std::vector<float> tmp(numSamples);

    biquadLP(input, tmp.data(), numSamples, lp1_, b0_lp, b1_lp, b2_lp, a1_lp, a2_lp);
    biquadLP(tmp.data(), lowOut, numSamples, lp2_, b0_lp, b1_lp, b2_lp, a1_lp, a2_lp);

    biquadHP(input, tmp.data(), numSamples, hp1_, b0_hp, b1_hp, b2_hp, a1_hp, a2_hp);
    biquadHP(tmp.data(), highOut, numSamples, hp2_, b0_hp, b1_hp, b2_hp, a1_hp, a2_hp);
}

}
