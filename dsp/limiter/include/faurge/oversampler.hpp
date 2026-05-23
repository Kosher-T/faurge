#pragma once

#include "lim_types.hpp"
#include "biquad.hpp"

#include <functional>
#include <vector>

namespace faurge {

class Oversampler {
public:
    using ProcessFn = std::function<LimiterResult(std::vector<float>&, int)>;

    Oversampler();

    LimiterResult process(std::vector<float>& audio, int sampleRate,
                          int oversamplingFactor, ProcessFn fn);

private:
    BiquadState upState1_, upState2_;
    BiquadState downState1_, downState2_;
    BiquadCoeffs upCoeffs_, downCoeffs_;
    int currentFactor_ = 1;

    void configureFilters(int inputSampleRate, int factor);
};

} // namespace faurge
