#include "faurge/sibilance_detector.hpp"

#include <algorithm>
#include <cmath>

namespace faurge {

SibilanceDetector::SibilanceDetector() {
    reset();
}

void SibilanceDetector::reset() {
    bandpass_.reset();
    configured_ = false;
}

void SibilanceDetector::configure(float centerFreqHz, float bandwidthHz,
                                   int sampleRate) {
    float clampedCenter = std::clamp(centerFreqHz, 20.0f, 20000.0f);
    float clampedBW = std::max(bandwidthHz, 50.0f);
    float q = clampedCenter / clampedBW;
    bandpass_.setFilter(clampedCenter, q, sampleRate);
    configured_ = true;
}

float SibilanceDetector::processSample(float input) {
    return bandpass_.processSample(input);
}

void SibilanceDetector::processBlock(const float* audio, float* detection,
                                      size_t n) {
    for (size_t i = 0; i < n; ++i) {
        detection[i] = bandpass_.processSample(audio[i]);
    }
}

} // namespace faurge
