#pragma once

#include "declipper_types.hpp"

#include <vector>

namespace faurge {

class PostFilter {
public:
    explicit PostFilter(const DeclipConfig& config);

    void apply(float* audio, size_t numSamples,
               const std::vector<ClipRegion>& regions,
               int sampleRate) const;

private:
    DeclipConfig cfg_;

    void crossfadeBlend(float* audio, size_t numSamples, const ClipRegion& region) const;
    void antiAliasFilter(float* audio, size_t start, size_t end, int sampleRate) const;
    void dcBlock(float* audio, size_t start, size_t end, int sampleRate) const;
};

} // namespace faurge
