// ═══════════════════════════════════════════════════════════════
// Faurge Specialist — De-Clipper: Post-Processing Filters
// ═══════════════════════════════════════════════════════════════
// Smooths the reconstruction output:
//   1. Raised-cosine crossfade at clip boundaries
//   2. 2nd-order Butterworth low-pass (anti-alias)
//   3. 1st-order high-pass DC blocker
// ═══════════════════════════════════════════════════════════════
#include "faurge/post_filter.hpp"

#include <algorithm>
#include <cmath>

namespace faurge {

static constexpr float PI = 3.14159265358979323846f;

PostFilter::PostFilter(const DeclipConfig& config) : cfg_(config) {}

// ── Public entry point ───────────────────────────────────────
void PostFilter::apply(float* audio, size_t numSamples,
                       const std::vector<ClipRegion>& regions,
                       int sampleRate) const {
    for (const auto& region : regions) {
        // 1. Crossfade at boundaries
        crossfadeBlend(audio, numSamples, region);

        // 2. Anti-alias the reconstructed region
        size_t margin = static_cast<size_t>(cfg_.crossfadeWidth);
        size_t start = (region.startSample > margin)
                           ? region.startSample - margin : 0;
        size_t end = std::min(region.endSample + margin, numSamples - 1);

        if (cfg_.enableAntiAlias) {
            antiAliasFilter(audio, start, end, sampleRate);
        }

        // 3. DC block the reconstructed region
        dcBlock(audio, start, end, sampleRate);
    }
}

// ── Raised-cosine crossfade ──────────────────────────────────
// Blends the original (clipped) boundary samples with the
// reconstructed samples over cfg_.crossfadeWidth samples to
// eliminate discontinuity artefacts at the junctions.
//
// At the left boundary:
//   for i in [0, width):
//     w = 0.5 * (1 - cos(π * i / width))   // ramps 0→1
//     audio[start-width+i] = (1-w)*original + w*reconstructed
//
// Since we've already overwritten the clipped region, the
// "original" is the unmodified data just outside the clip.
// We only need to smooth the transition zone.
void PostFilter::crossfadeBlend(float* audio, size_t numSamples,
                                const ClipRegion& region) const {
    int width = cfg_.crossfadeWidth;
    if (width <= 0) return;

    // Left boundary crossfade
    {
        int fadeStart = static_cast<int>(region.startSample) - width;
        if (fadeStart < 0) fadeStart = 0;
        int fadeEnd = static_cast<int>(region.startSample);
        int fadeLen = fadeEnd - fadeStart;
        if (fadeLen > 0) {
            for (int i = 0; i < fadeLen; ++i) {
                float t = static_cast<float>(i) / static_cast<float>(fadeLen);
                // At i=0: keep original; at i=fadeLen-1: mostly reconstructed
                // But since the original IS the audio before the clip, we
                // just smooth the boundary.  Apply a subtle gain ramp to
                // avoid abrupt slope changes.
                float w = 0.5f * (1.0f - std::cos(PI * t));
                int idx = fadeStart + i;
                // Blend towards the first reconstructed sample
                float target = audio[region.startSample];
                audio[idx] = audio[idx] * (1.0f - w * 0.1f) + target * (w * 0.1f);
            }
        }
    }

    // Right boundary crossfade
    {
        int fadeStart = static_cast<int>(region.endSample) + 1;
        int fadeEnd = fadeStart + width;
        if (fadeEnd > static_cast<int>(numSamples)) fadeEnd = static_cast<int>(numSamples);
        int fadeLen = fadeEnd - fadeStart;
        if (fadeLen > 0) {
            for (int i = 0; i < fadeLen; ++i) {
                float t = static_cast<float>(i) / static_cast<float>(fadeLen);
                float w = 0.5f * (1.0f + std::cos(PI * t));
                int idx = fadeStart + i;
                float target = audio[region.endSample];
                audio[idx] = audio[idx] * (1.0f - w * 0.1f) + target * (w * 0.1f);
            }
        }
    }
}

// ── Butterworth anti-alias filter ────────────────────────────
// 2nd-order Butterworth low-pass applied only to the
// reconstructed region to suppress any synthesis artefacts.
// Cutoff = sampleRate / 4 (Nyquist / 2).
void PostFilter::antiAliasFilter(float* audio, size_t start, size_t end,
                                 int sampleRate) const {
    if (start >= end) return;

    // Butterworth 2nd order LPF coefficients
    float fc = static_cast<float>(sampleRate) / 4.0f;
    float wc = std::tan(PI * fc / static_cast<float>(sampleRate));
    float wc2 = wc * wc;
    float sqrt2 = 1.41421356f;
    float k = sqrt2 * wc;
    float norm = 1.0f / (1.0f + k + wc2);

    float b0 = wc2 * norm;
    float b1 = 2.0f * b0;
    float b2 = b0;
    float a1 = 2.0f * (wc2 - 1.0f) * norm;
    float a2 = (1.0f - k + wc2) * norm;

    // Direct Form II Transposed
    float z1 = 0.0f, z2 = 0.0f;

    // Forward pass
    for (size_t i = start; i <= end; ++i) {
        float in = audio[i];
        float out = b0 * in + z1;
        z1 = b1 * in - a1 * out + z2;
        z2 = b2 * in - a2 * out;
        audio[i] = out;
    }

    // Backward pass (zero-phase filtering — eliminates phase distortion)
    z1 = z2 = 0.0f;
    for (size_t i = end + 1; i > start; --i) {
        size_t idx = i - 1;
        float in = audio[idx];
        float out = b0 * in + z1;
        z1 = b1 * in - a1 * out + z2;
        z2 = b2 * in - a2 * out;
        audio[idx] = out;
    }
}

// ── DC blocker ───────────────────────────────────────────────
// 1st-order high-pass filter to remove any DC offset introduced
// by the reconstruction.  Cutoff at cfg_.dcBlockFreqHz (10 Hz).
//
// Transfer function:  H(z) = (1 - z⁻¹) / (1 - α·z⁻¹)
// where α = 1 - 2π·fc/fs
void PostFilter::dcBlock(float* audio, size_t start, size_t end,
                         int sampleRate) const {
    if (start >= end) return;

    float fc = cfg_.dcBlockFreqHz;
    float alpha = 1.0f - (2.0f * PI * fc / static_cast<float>(sampleRate));
    // Clamp alpha to valid range
    alpha = std::max(0.0f, std::min(alpha, 0.9999f));

    float xPrev = audio[start];
    float yPrev = audio[start];

    for (size_t i = start + 1; i <= end; ++i) {
        float x = audio[i];
        float y = alpha * yPrev + x - xPrev;
        xPrev = x;
        yPrev = y;
        audio[i] = y;
    }
}

}  // namespace faurge
