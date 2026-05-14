#pragma once

#include <vector>

namespace faurge {

class NoiseEstimator {
public:
    NoiseEstimator() = default;

    float estimateNoiseFloorDb(const float* audio, size_t numSamples,
                               int sampleRate);

    float estimateSnrDb(const float* audio, size_t numSamples,
                        int sampleRate);

private:
    std::vector<float> noiseFloorHistory_;
    size_t historyPos_ = 0;

    float computeRmsDb(const float* audio, size_t numSamples) const;
};

}  // namespace faurge
