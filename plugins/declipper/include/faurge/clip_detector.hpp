// ═══════════════════════════════════════════════════════════════
// Faurge Specialist — De-Clipper: Stage 1 — Clip Detection
// ═══════════════════════════════════════════════════════════════
// Multi-mode clip detection engine.
//   • Hard-clip:  consecutive samples at ±threshold
//   • Soft-clip:  derivative discontinuity detection
//   • ISP:        inter-sample peak detection via 4× oversampling
// ═══════════════════════════════════════════════════════════════
#pragma once

#include "clip_region.hpp"
#include <vector>

namespace faurge {

class ClipDetector {
public:
    explicit ClipDetector(const DeclipConfig& config);

    // Run full detection pipeline on a mono audio buffer.
    // Returns a ClipReport with all detected regions, boundaries filled.
    ClipReport detect(const float* audio, size_t numSamples) const;

    // Assign severity based on region length
    static ClipSeverity classifySeverity(size_t length);

private:
    DeclipConfig cfg_;

    // Stage 1a: Hard-clip detection — consecutive samples at ceiling
    std::vector<ClipRegion> detectHardClips(const float* audio, size_t n) const;

    // Stage 1b: Soft-clip detection — second-derivative discontinuity
    std::vector<ClipRegion> detectSoftClips(const float* audio, size_t n) const;

    // Merge overlapping / close regions and classify severity
    void mergeAndClassify(std::vector<ClipRegion>& regions) const;

    // Fill boundary anchor samples from the original audio
    void fillAnchors(std::vector<ClipRegion>& regions,
                     const float* audio, size_t n) const;
};

}  // namespace faurge
