#pragma once

#include "declipper_types.hpp"

#include <vector>

namespace faurge {

class Reconstructor {
public:
    explicit Reconstructor(const DeclipConfig& config);
    void reconstruct(float* audio, size_t numSamples, std::vector<ClipRegion>& regions) const;

private:
    DeclipConfig cfg_;

    void hermiteReconstruct(float* audio, ClipRegion& region) const;
    void akimaReconstruct(float* audio, ClipRegion& region) const;
    void arReconstruct(float* audio, size_t numSamples, ClipRegion& region) const;

    float estimatePeak(const ClipRegion& region) const;
    float constrainSample(float value, float peakEst, ClipPolarity polarity) const;
};

} // namespace faurge
