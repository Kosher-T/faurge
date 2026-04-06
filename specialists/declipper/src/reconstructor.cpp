// ═══════════════════════════════════════════════════════════════
// Faurge Specialist — De-Clipper: Waveform Reconstruction
// ═══════════════════════════════════════════════════════════════
// Three-tier reconstruction:
//   1. Cubic Hermite Spline   — short clips (≤16 samples)
//   2. Akima Sub-Spline       — medium clips (17–64 samples)
//   3. Burg AR Extrapolation  — long / sustained clips (>64)
//
// All methods: polarity-aware, peak-constrained, C1-continuous.
// ═══════════════════════════════════════════════════════════════
#include "faurge/reconstructor.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>

namespace faurge {

// ── Constructor ──────────────────────────────────────────────
Reconstructor::Reconstructor(const DeclipConfig& config) : cfg_(config) {}

// ── Public entry point ───────────────────────────────────────
void Reconstructor::reconstruct(float* audio, size_t numSamples,
                                std::vector<ClipRegion>& regions) const {
    for (auto& region : regions) {
        // Estimate true peak amplitude from boundary slopes
        region.estimatedPeakAmplitude = estimatePeak(region);

        size_t len = region.length();
        if (static_cast<int>(len) <= cfg_.hermiteMaxLen) {
            hermiteReconstruct(audio, region);
        } else if (static_cast<int>(len) <= cfg_.akimaMaxLen) {
            akimaReconstruct(audio, region);
        } else {
            arReconstruct(audio, numSamples, region);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Strategy 1: Cubic Hermite Spline
// ═══════════════════════════════════════════════════════════════
// Uses the Hermite basis functions H00, H10, H01, H11 to
// interpolate a C1-continuous curve between two boundary points.
//
// The boundary points are the last anchor before the clip and
// the first anchor after.  Slopes are estimated via centered
// finite difference on the anchor arrays.
//
// Hermite basis (for t ∈ [0,1]):
//   H00(t) =  2t³ - 3t² + 1
//   H10(t) =   t³ - 2t² + t
//   H01(t) = -2t³ + 3t²
//   H11(t) =   t³ -  t²
//
// p(t) = H00·p0 + H10·m0 + H01·p1 + H11·m1
// ═══════════════════════════════════════════════════════════════
void Reconstructor::hermiteReconstruct(float* audio,
                                       ClipRegion& region) const {
    const auto& before = region.anchorsBefore;
    const auto& after  = region.anchorsAfter;

    // We need at least 2 anchors on each side for slope estimation
    if (before.size() < 2 || after.size() < 2) {
        // Fallback: linear interpolation if not enough context
        float p0 = before.empty() ? 0.0f : before.back();
        float p1 = after.empty()  ? 0.0f : after.front();
        size_t len = region.length();
        for (size_t i = 0; i < len; ++i) {
            float t = static_cast<float>(i + 1) / static_cast<float>(len + 1);
            audio[region.startSample + i] = p0 + t * (p1 - p0);
        }
        return;
    }

    // Boundary values
    float p0 = before.back();
    float p1 = after.front();

    // Slope estimation via finite difference
    // m0 = slope entering the clip (from anchor side)
    float m0 = 0.0f;
    {
        size_t nb = before.size();
        if (nb >= 3) {
            // Centered difference on last 3 anchors
            m0 = (before[nb - 1] - before[nb - 3]) / 2.0f;
        } else {
            m0 = before[nb - 1] - before[nb - 2];
        }
    }

    // m1 = slope exiting the clip (into anchor side)
    float m1 = 0.0f;
    {
        size_t na = after.size();
        if (na >= 3) {
            m1 = (after[2] - after[0]) / 2.0f;
        } else {
            m1 = after[1] - after[0];
        }
    }

    // Scale slopes by the interval length for proper parametrisation
    float intervalLen = static_cast<float>(region.length() + 1);
    m0 *= intervalLen;
    m1 *= intervalLen;

    // Interpolate
    size_t len = region.length();
    for (size_t i = 0; i < len; ++i) {
        float t = static_cast<float>(i + 1) / static_cast<float>(len + 1);
        float t2 = t * t;
        float t3 = t2 * t;

        // Hermite basis functions
        float h00 =  2.0f * t3 - 3.0f * t2 + 1.0f;
        float h10 =         t3 - 2.0f * t2 + t;
        float h01 = -2.0f * t3 + 3.0f * t2;
        float h11 =         t3 -        t2;

        float value = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;

        // Constrain to estimated peak
        audio[region.startSample + i] =
            constrainSample(value, region.estimatedPeakAmplitude,
                            region.polarity);
    }
}

// ═══════════════════════════════════════════════════════════════
// Strategy 2: Akima Sub-Spline
// ═══════════════════════════════════════════════════════════════
// Akima splines use a weighted average of surrounding slopes to
// avoid the oscillation problems (Runge phenomenon) that natural
// cubic splines exhibit on longer intervals.
//
// We construct a set of knot points:
//   [anchors_before..., threshold, ..., threshold, anchors_after...]
// and fit Akima piecewise cubics through them.
// ═══════════════════════════════════════════════════════════════
void Reconstructor::akimaReconstruct(float* audio,
                                     ClipRegion& region) const {
    // Build knot sequence: before anchors + clip region + after anchors
    std::vector<float> knots;
    std::vector<float> xpos;  // x positions (sample-normalised)

    const auto& before = region.anchorsBefore;
    const auto& after  = region.anchorsAfter;

    int anchorN = static_cast<int>(before.size());
    int clipLen = static_cast<int>(region.length());
    int afterN  = static_cast<int>(after.size());
    int totalKnots = anchorN + 2 + afterN;  // +2 for clip start/end boundary

    knots.resize(totalKnots);
    xpos.resize(totalKnots);

    int idx = 0;
    // Before anchors
    for (int i = 0; i < anchorN; ++i) {
        knots[idx] = before[i];
        xpos[idx]  = static_cast<float>(i - anchorN);
        ++idx;
    }
    // Clip start boundary (last good sample = p0)
    knots[idx] = before.empty() ? audio[region.startSample] : before.back();
    xpos[idx]  = 0.0f;
    ++idx;
    // Clip end boundary
    knots[idx] = after.empty() ? audio[region.endSample] : after.front();
    xpos[idx]  = static_cast<float>(clipLen + 1);
    ++idx;
    // After anchors
    for (int i = 0; i < afterN; ++i) {
        knots[idx] = after[i];
        xpos[idx]  = static_cast<float>(clipLen + 2 + i);
        ++idx;
    }

    // Compute Akima slopes at each knot
    int nk = static_cast<int>(knots.size());
    std::vector<float> slopes(nk, 0.0f);

    // Divided differences
    std::vector<float> dd(nk - 1, 0.0f);
    for (int i = 0; i < nk - 1; ++i) {
        float dx = xpos[i + 1] - xpos[i];
        dd[i] = (dx != 0.0f) ? (knots[i + 1] - knots[i]) / dx : 0.0f;
    }

    // Akima slope formula:
    // s_i = (|d_{i+1} - d_i| * d_{i-1} + |d_{i-1} - d_{i-2}| * d_i)
    //       / (|d_{i+1} - d_i| + |d_{i-1} - d_{i-2}|)
    for (int i = 0; i < nk; ++i) {
        if (i < 2 || i >= nk - 2) {
            // At boundaries, use simple difference
            if (i < nk - 1) slopes[i] = dd[std::min(i, nk - 2)];
            else slopes[i] = dd[nk - 2];
        } else {
            float w1 = std::fabs(dd[i] - dd[i - 1]);
            float w2 = std::fabs(dd[i - 2] - dd[i - 1]);
            // Handle edge case when weights are both zero
            // (locally linear section)
            float totalW = w1 + w2;
            if (totalW < 1e-12f) {
                slopes[i] = 0.5f * (dd[i - 1] + dd[i]);
            } else {
                // Note: Akima's original formula (corrected):
                // w1 = |d_{i+1} - d_i|, w2 = |d_{i-1} - d_{i-2}|
                // slope = (w1 * d_{i-1} + w2 * d_i) / (w1 + w2)
                float wa = std::fabs(dd[std::min(i, nk - 2)] - dd[i - 1]);
                float wb = std::fabs(dd[i - 2] - dd[std::max(i - 3, 0)]);
                float tw = wa + wb;
                if (tw < 1e-12f) {
                    slopes[i] = 0.5f * (dd[i - 1] + dd[std::min(i, nk - 2)]);
                } else {
                    slopes[i] = (wa * dd[i - 1] + wb * dd[std::min(i, nk - 2)]) / tw;
                }
            }
        }
    }

    // Interpolate clipped samples using piecewise Hermite with Akima slopes
    // We interpolate between knot[anchorN] (x=0) and knot[anchorN+1] (x=clipLen+1)
    int knotLeft  = anchorN;
    int knotRight = anchorN + 1;
    float x0 = xpos[knotLeft];
    float x1 = xpos[knotRight];
    float dx = x1 - x0;
    float p0 = knots[knotLeft];
    float p1 = knots[knotRight];
    float s0 = slopes[knotLeft] * dx;
    float s1 = slopes[knotRight] * dx;

    for (int i = 0; i < clipLen; ++i) {
        float x = static_cast<float>(i + 1);
        float t = (x - x0) / dx;
        float t2 = t * t;
        float t3 = t2 * t;

        float h00 =  2.0f * t3 - 3.0f * t2 + 1.0f;
        float h10 =         t3 - 2.0f * t2 + t;
        float h01 = -2.0f * t3 + 3.0f * t2;
        float h11 =         t3 -        t2;

        float value = h00 * p0 + h10 * s0 + h01 * p1 + h11 * s1;

        audio[region.startSample + i] =
            constrainSample(value, region.estimatedPeakAmplitude,
                            region.polarity);
    }
}

// ═══════════════════════════════════════════════════════════════
// Strategy 3: Burg AR Extrapolation
// ═══════════════════════════════════════════════════════════════
// For sustained clips (>64 samples) where spline interpolation
// would diverge wildly, we use autoregressive modelling.
//
// 1. Fit an AR model (order ~14) to the pre-clip context using
//    the Burg method (maximises entropy, very stable).
// 2. Forward-predict from the left boundary.
// 3. Fit another AR model to the post-clip context.
// 4. Backward-predict from the right boundary.
// 5. Blend forward and backward predictions with a raised-cosine
//    crossfade through the middle of the gap.
// ═══════════════════════════════════════════════════════════════
void Reconstructor::arReconstruct(float* audio, size_t numSamples,
                                  ClipRegion& region) const {
    int order = cfg_.arModelOrder;
    int clipLen = static_cast<int>(region.length());

    // Context size — we want at least 4× the AR order on each side
    int contextLen = order * 4;

    // ── Burg method to estimate AR coefficients ──────────────
    auto burgFit = [](const std::vector<float>& data, int p)
                       -> std::vector<float> {
        int n = static_cast<int>(data.size());
        if (n <= p) {
            return std::vector<float>(p, 0.0f);
        }

        std::vector<float> a(p + 1, 0.0f);
        a[0] = 1.0f;

        std::vector<float> ef(data.begin(), data.end());  // forward errors
        std::vector<float> eb(data.begin(), data.end());  // backward errors

        for (int m = 0; m < p; ++m) {
            // Compute reflection coefficient
            float num = 0.0f, den = 0.0f;
            for (int j = m + 1; j < n; ++j) {
                num += ef[j] * eb[j - 1];
                den += ef[j] * ef[j] + eb[j - 1] * eb[j - 1];
            }
            float km = (den > 1e-30f) ? (-2.0f * num / den) : 0.0f;

            // Update AR coefficients (Levinson-Durbin style)
            std::vector<float> aNew(p + 1, 0.0f);
            aNew[0] = 1.0f;
            for (int i = 1; i <= m; ++i) {
                aNew[i] = a[i] + km * a[m + 1 - i];
            }
            aNew[m + 1] = km;
            a = aNew;

            // Update prediction errors
            std::vector<float> efNew(n, 0.0f);
            for (int j = m + 1; j < n; ++j) {
                efNew[j] = ef[j] + km * eb[j - 1];
            }
            for (int j = m + 1; j < n; ++j) {
                eb[j] = eb[j - 1] + km * ef[j];
            }
            ef = efNew;
        }

        // Return coefficients a[1..p] (a[0]==1 is implicit)
        return std::vector<float>(a.begin() + 1, a.end());
    };

    // ── Gather forward context (samples before the clip) ─────
    std::vector<float> fwdContext;
    {
        int start = std::max(0, static_cast<int>(region.startSample) - contextLen);
        for (int i = start; i < static_cast<int>(region.startSample); ++i) {
            fwdContext.push_back(audio[i]);
        }
    }

    // ── Gather backward context (samples after the clip) ─────
    std::vector<float> bwdContext;
    {
        size_t start = region.endSample + 1;
        size_t end   = std::min(numSamples, start + contextLen);
        for (size_t i = start; i < end; ++i) {
            bwdContext.push_back(audio[i]);
        }
    }

    // ── Forward prediction ───────────────────────────────────
    auto fwdCoeffs = burgFit(fwdContext, order);
    std::vector<float> fwdPred(clipLen, 0.0f);
    {
        // Seed the predictor with the tail of fwdContext
        std::vector<float> buf(fwdContext);
        for (int i = 0; i < clipLen; ++i) {
            float val = 0.0f;
            int bLen = static_cast<int>(buf.size());
            for (int k = 0; k < order && k < bLen; ++k) {
                val -= fwdCoeffs[k] * buf[bLen - 1 - k];
            }
            fwdPred[i] = val;
            buf.push_back(val);
        }
    }

    // ── Backward prediction ──────────────────────────────────
    // Reverse the backward context, fit AR, predict, then reverse result
    std::vector<float> bwdContextRev(bwdContext.rbegin(), bwdContext.rend());
    auto bwdCoeffs = burgFit(bwdContextRev, order);
    std::vector<float> bwdPred(clipLen, 0.0f);
    {
        std::vector<float> buf(bwdContextRev);
        for (int i = 0; i < clipLen; ++i) {
            float val = 0.0f;
            int bLen = static_cast<int>(buf.size());
            for (int k = 0; k < order && k < bLen; ++k) {
                val -= bwdCoeffs[k] * buf[bLen - 1 - k];
            }
            bwdPred[i] = val;
            buf.push_back(val);
        }
        // Reverse to get forward-time order
        std::reverse(bwdPred.begin(), bwdPred.end());
    }

    // ── Blend forward and backward with raised-cosine crossfade ──
    for (int i = 0; i < clipLen; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(std::max(clipLen - 1, 1));
        // Raised cosine: 1.0 at start (favour forward), 0.0 at end (favour backward)
        float wFwd = 0.5f * (1.0f + std::cos(t * 3.14159265f));
        float wBwd = 1.0f - wFwd;

        float blended = wFwd * fwdPred[i] + wBwd * bwdPred[i];

        audio[region.startSample + i] =
            constrainSample(blended, region.estimatedPeakAmplitude,
                            region.polarity);
    }
}

// ── Peak estimation ──────────────────────────────────────────
// Uses a parabolic fit through the last few anchors before the
// clip to estimate how high the waveform would have peaked had
// it not been chopped.
float Reconstructor::estimatePeak(const ClipRegion& region) const {
    const auto& before = region.anchorsBefore;
    const auto& after  = region.anchorsAfter;

    // Compute slopes at the clip boundaries
    float slopeBefore = 0.0f;
    if (before.size() >= 2) {
        slopeBefore = before.back() - before[before.size() - 2];
    }

    float slopeAfter = 0.0f;
    if (after.size() >= 2) {
        slopeAfter = after[1] - after[0];
    }

    // The peak occurs roughly where the slope crosses zero.
    // For a symmetric clip, the slope magnitudes should be roughly
    // equal and opposite.  We estimate:
    //   peak ≈ threshold + |slope| × (clipLength / 4)
    // Clamped by the config overshoot limit.
    float avgSlope = (std::fabs(slopeBefore) + std::fabs(slopeAfter)) / 2.0f;
    float clipLen  = static_cast<float>(region.length());
    float rawPeak  = cfg_.clipThreshold + avgSlope * clipLen * 0.25f;

    // Clamp to overshoot limit
    float maxPeak = cfg_.clipThreshold * cfg_.peakOvershoot;
    return std::min(rawPeak, maxPeak);
}

// ── Constrain sample ─────────────────────────────────────────
float Reconstructor::constrainSample(float value, float peakEst,
                                     ClipPolarity polarity) const {
    if (polarity == ClipPolarity::Positive) {
        // Must be above threshold (we're drawing the peak higher)
        // but not above estimated peak
        return std::min(value, peakEst);
    } else {
        return std::max(value, -peakEst);
    }
}

}  // namespace faurge
