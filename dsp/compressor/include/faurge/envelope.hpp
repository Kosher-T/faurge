#pragma once

#include "comp_types.hpp"

#include <cstddef>
#include <vector>

namespace faurge {

class EnvelopeFollower {
public:
    EnvelopeFollower();

    void reset();

    float processSample(float input, DetectorType type,
                        float attackMs, float releaseMs, int sampleRate);

    void processBlock(const float* audio, float* envelope, size_t n,
                      DetectorType type, float attackMs, float releaseMs,
                      int sampleRate);

private:
    float envelope_;
    float sampleRate_;

    static float rcAlpha(float tauMs, int sampleRate);
};

}
