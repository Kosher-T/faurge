// ═══════════════════════════════════════════════════════════════
// Faurge Specialist — De-Clipper: Stage 4 — Post-Processing
// ═══════════════════════════════════════════════════════════════
// Smooths reconstruction artefacts:
//   • Raised-cosine crossfade at clip boundaries
//   • 2nd-order Butterworth anti-alias on reconstructed regions
//   • 1st-order DC-blocking high-pass at 10 Hz
// ═══════════════════════════════════════════════════════════════
#pragma once

#include "clip_region.hpp"
#include <vector>

namespace faurge {

class PostFilter {
public:
    explicit PostFilter(const DeclipConfig& config);

    // Apply all post-processing to the reconstructed audio in-place.
    void apply(float* audio, size_t numSamples,
               const std::vector<ClipRegion>& regions,
               int sampleRate) const;

    // ── Individual filters (public for testing) ──────────────

    // Raised-cosine crossfade blending at clip boundaries
    void crossfadeBlend(float* audio, size_t numSamples,
                        const ClipRegion& region) const;

    // 2nd-order Butterworth low-pass on a sub-range
    void antiAliasFilter(float* audio, size_t start, size_t end,
                         int sampleRate) const;

    // 1st-order high-pass DC blocker on a sub-range
    void dcBlock(float* audio, size_t start, size_t end,
                 int sampleRate) const;

private:
    DeclipConfig cfg_;
};

}  // namespace faurge
