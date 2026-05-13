// ═══════════════════════════════════════════════════════════════
// Faurge Specialist — De-Clipper: Stage 3 — Waveform Reconstruction
// ═══════════════════════════════════════════════════════════════
// Three-tier polynomial / AR reconstruction engine:
//   1. Cubic Hermite Spline   (short clips, ≤16 samples)
//   2. Akima Spline           (medium clips, 17–64 samples)
//   3. Burg AR Extrapolation  (long clips, >64 samples)
// ═══════════════════════════════════════════════════════════════
#pragma once

#include "clip_region.hpp"
#include <vector>

namespace faurge {

class Reconstructor {
public:
    explicit Reconstructor(const DeclipConfig& config);

    // Reconstruct all clipped regions in-place.
    // The audio buffer is modified directly.
    void reconstruct(float* audio, size_t numSamples,
                     std::vector<ClipRegion>& regions) const;

    // ── Individual strategies (public for unit testing) ──────

    // Cubic Hermite spline interpolation between boundary anchors.
    // Guarantees C1 continuity at the junctions.
    void hermiteReconstruct(float* audio, ClipRegion& region) const;

    // Akima sub-spline — piecewise cubic with Akima slope estimation.
    // More robust than natural cubic for longer intervals.
    void akimaReconstruct(float* audio, ClipRegion& region) const;

    // Burg-method autoregressive forward-backward prediction.
    // For sustained clips where spline methods diverge.
    void arReconstruct(float* audio, size_t numSamples,
                       ClipRegion& region) const;

private:
    DeclipConfig cfg_;

    // Estimate the true peak amplitude from boundary slopes
    float estimatePeak(const ClipRegion& region) const;

    // Constrain reconstructed value to stay within estimated peak
    float constrainSample(float value, float peakEst,
                          ClipPolarity polarity) const;
};

}  // namespace faurge
