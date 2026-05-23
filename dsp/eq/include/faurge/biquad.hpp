#pragma once

#include "eq_types.hpp"

#include <cstddef>
#include <vector>

namespace faurge {

struct BiquadCoeffs {
    float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f;
    float a1 = 0.0f, a2 = 0.0f;
};

struct BiquadState {
    float x1 = 0.0f, x2 = 0.0f, y1 = 0.0f, y2 = 0.0f;
};

class Biquad {
public:
    Biquad();

    void reset();
    void setFilter(FilterType type, float freq, float gainDb, float q, int sampleRate);
    void process(std::vector<float>& audio);

private:
    BiquadState state_;
    BiquadCoeffs coeffs_;

    void processRaw(const float* in, float* out, size_t n);

    static BiquadCoeffs designPeak(float freq, float gainDb, float q, int sampleRate);
    static BiquadCoeffs designLowShelf(float freq, float gainDb, float q, int sampleRate);
    static BiquadCoeffs designHighShelf(float freq, float gainDb, float q, int sampleRate);
    static BiquadCoeffs designHighpass(float freq, float q, int sampleRate);
    static BiquadCoeffs designLowpass(float freq, float q, int sampleRate);
    static BiquadCoeffs designBandpass(float freq, float q, int sampleRate);
    static BiquadCoeffs designNotch(float freq, float q, int sampleRate);
};

}
