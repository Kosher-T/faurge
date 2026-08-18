#pragma once

#include "declipper_types.hpp"

#include <vector>

namespace faurge {

class ClipDetector {
public:
    explicit ClipDetector(const DeclipConfig& config);
    ClipReport detect(const float* audio, size_t numSamples) const;
    static ClipSeverity classifySeverity(size_t length);

private:
    DeclipConfig cfg_;

    std::vector<ClipRegion> detectHardClips(const float* audio, size_t n) const;
    std::vector<ClipRegion> detectSoftClips(const float* audio, size_t n) const;
    void mergeAndClassify(std::vector<ClipRegion>& regions) const;
    void fillAnchors(std::vector<ClipRegion>& regions, const float* audio, size_t n) const;
};

} // namespace faurge
