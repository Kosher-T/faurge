#pragma once

#include <cstddef>

namespace faurge {

class CrossoverFilter {
public:
    CrossoverFilter();

    void reset();

    void process(const float* input, float* lowOut, float* highOut,
                 size_t numSamples, int sampleRate, float crossoverHz);

private:
    struct BiquadState { float x1 = 0.0f, x2 = 0.0f, y1 = 0.0f, y2 = 0.0f; };

    BiquadState lp1_, lp2_;
    BiquadState hp1_, hp2_;

    static void biquadLP(const float* in, float* out, size_t n,
                         BiquadState& s, float b0, float b1, float b2,
                         float a1, float a2);
    static void biquadHP(const float* in, float* out, size_t n,
                         BiquadState& s, float b0, float b1, float b2,
                         float a1, float a2);
};

}
