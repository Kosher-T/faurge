#include "faurge/waveshaper.hpp"

#include <cmath>

namespace faurge {

float saturateTube(float x) {
    float g = 1.3f * x;
    return g / (1.0f + std::abs(g));
}

float saturateTape(float x) {
    return std::tanh(x);
}

float saturateDiode(float x) {
    if (x >= 0.0f) return 1.0f - std::exp(-x);
    return std::exp(x) - 1.0f;
}

float saturateAsymmetric(float x) {
    float g = (x >= 0.0f) ? (1.5f * x) : (0.7f * x);
    return g / (1.0f + std::abs(g));
}

WaveshaperFn getWaveshaper(int type) {
    switch (type) {
        case 0:  return saturateTube;
        case 1:  return saturateTape;
        case 2:  return saturateDiode;
        case 3:  return saturateAsymmetric;
        default: return saturateTape;
    }
}

} // namespace faurge
