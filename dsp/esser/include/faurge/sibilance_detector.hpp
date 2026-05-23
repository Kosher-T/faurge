#pragma once

#include "biquad.hpp"

#include <cstddef>

namespace faurge {

class SibilanceDetector {
public:
    SibilanceDetector();

    void reset();
    void configure(float centerFreqHz, float bandwidthHz, int sampleRate);
    float processSample(float input);
    void processBlock(const float* audio, float* detection, size_t n);

    bool isConfigured() const { return configured_; }

private:
    Biquad bandpass_;
    bool configured_ = false;
};

} // namespace faurge
