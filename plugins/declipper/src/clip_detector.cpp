// ═══════════════════════════════════════════════════════════════
// Faurge Specialist — De-Clipper: Clip Detection Implementation
// ═══════════════════════════════════════════════════════════════
#include "faurge/clip_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace faurge {

// ── Constructor ──────────────────────────────────────────────
ClipDetector::ClipDetector(const DeclipConfig& config) : cfg_(config) {}

// ── Public entry point ───────────────────────────────────────
ClipReport ClipDetector::detect(const float* audio, size_t numSamples) const {
    // Stage 1a — hard clips
    auto regions = detectHardClips(audio, numSamples);

    // Stage 1b — soft clips (optional)
    if (cfg_.detectSoftClip) {
        auto soft = detectSoftClips(audio, numSamples);
        regions.insert(regions.end(), soft.begin(), soft.end());
    }

    // Merge overlapping / neighbouring regions and classify severity
    mergeAndClassify(regions);

    // Fill boundary anchors from original audio
    fillAnchors(regions, audio, numSamples);

    // Build report
    ClipReport report;
    report.regions      = std::move(regions);
    report.totalSamples = numSamples;
    report.totalClippedSamples = 0;
    for (const auto& r : report.regions) {
        report.totalClippedSamples += r.length();
    }
    report.percentClipped = numSamples > 0
        ? 100.0f * static_cast<float>(report.totalClippedSamples)
                  / static_cast<float>(numSamples)
        : 0.0f;

    return report;
}

// ── Hard-clip detection ──────────────────────────────────────
// A "hard clip" is a run of consecutive samples whose absolute
// value meets or exceeds the threshold (default 0.9999).
std::vector<ClipRegion> ClipDetector::detectHardClips(
        const float* audio, size_t n) const {
    std::vector<ClipRegion> regions;
    size_t i = 0;
    while (i < n) {
        float absVal = std::fabs(audio[i]);
        if (absVal >= cfg_.clipThreshold) {
            ClipRegion region;
            region.startSample = i;
            region.polarity = (audio[i] >= 0.0f)
                ? ClipPolarity::Positive
                : ClipPolarity::Negative;

            // Walk forward while clipped
            while (i < n && std::fabs(audio[i]) >= cfg_.clipThreshold) {
                ++i;
            }
            region.endSample = i - 1;

            // Only keep if long enough
            if (static_cast<int>(region.length()) >= cfg_.minClipLength) {
                regions.push_back(region);
            }
        } else {
            ++i;
        }
    }
    return regions;
}

// ── Soft-clip detection ──────────────────────────────────────
// Detects regions where the waveform was squashed by an analog
// limiter before digitisation.  The tell-tale sign is an abrupt
// drop-off in the second derivative while the amplitude is high.
//
// We look for: |sample| > 0.9  AND  second-derivative magnitude
// drops below the softClipDerivThr for multiple consecutive
// samples.  This catches tanh/sigmoid-style saturation.
std::vector<ClipRegion> ClipDetector::detectSoftClips(
        const float* audio, size_t n) const {
    std::vector<ClipRegion> regions;
    if (n < 4) return regions;

    const float highAmpThreshold = cfg_.clipThreshold * 0.9f;  // ~0.9

    size_t i = 2;
    while (i < n - 1) {
        float absVal = std::fabs(audio[i]);
        if (absVal > highAmpThreshold && absVal < cfg_.clipThreshold) {
            // Compute second derivative (curvature)
            float d2 = audio[i + 1] - 2.0f * audio[i] + audio[i - 1];
            float absd2 = std::fabs(d2);

            // Normal audio has non-trivial curvature.  Soft-clipped
            // regions go nearly flat (linear), so d2 drops to ~0.
            // We normalise by amplitude to make the threshold
            // independent of loudness.
            float normCurvature = (absVal > 1e-6f) ? (absd2 / absVal) : absd2;

            if (normCurvature < (1.0f - cfg_.softClipDerivThr) * 0.1f) {
                ClipRegion region;
                region.startSample = i;
                region.polarity = (audio[i] >= 0.0f)
                    ? ClipPolarity::Positive
                    : ClipPolarity::Negative;

                while (i < n - 1) {
                    float av = std::fabs(audio[i]);
                    if (av <= highAmpThreshold) break;
                    float dd = std::fabs(audio[i + 1] - 2.0f * audio[i]
                                         + audio[i - 1]);
                    float nc = (av > 1e-6f) ? (dd / av) : dd;
                    if (nc >= (1.0f - cfg_.softClipDerivThr) * 0.1f) break;
                    ++i;
                }
                region.endSample = i - 1;

                if (static_cast<int>(region.length()) >= cfg_.minClipLength) {
                    regions.push_back(region);
                }
            } else {
                ++i;
            }
        } else {
            ++i;
        }
    }
    return regions;
}

// ── Merge & Classify ─────────────────────────────────────────
// Merge regions separated by ≤ mergeGap samples, then classify.
void ClipDetector::mergeAndClassify(std::vector<ClipRegion>& regions) const {
    if (regions.empty()) return;

    // Sort by start sample
    std::sort(regions.begin(), regions.end(),
              [](const ClipRegion& a, const ClipRegion& b) {
                  return a.startSample < b.startSample;
              });

    // Merge overlapping or close regions
    std::vector<ClipRegion> merged;
    merged.push_back(regions[0]);
    for (size_t i = 1; i < regions.size(); ++i) {
        auto& prev = merged.back();
        const auto& cur = regions[i];

        // Overlapping or within mergeGap
        if (cur.startSample <= prev.endSample + 1 + cfg_.mergeGap) {
            prev.endSample = std::max(prev.endSample, cur.endSample);
            // Keep polarity of the larger sub-region
        } else {
            merged.push_back(cur);
        }
    }

    // Classify severity
    for (auto& r : merged) {
        r.severity = classifySeverity(r.length());
    }

    regions = std::move(merged);
}

// ── Fill boundary anchors ────────────────────────────────────
void ClipDetector::fillAnchors(std::vector<ClipRegion>& regions,
                               const float* audio, size_t n) const {
    for (auto& r : regions) {
        r.anchorsBefore.clear();
        r.anchorsAfter.clear();

        // Anchors before the clip
        int startAnchor = static_cast<int>(r.startSample) - cfg_.anchorSize;
        if (startAnchor < 0) startAnchor = 0;
        for (int j = startAnchor; j < static_cast<int>(r.startSample); ++j) {
            r.anchorsBefore.push_back(audio[j]);
        }

        // Anchors after the clip
        size_t endAnchor = r.endSample + 1 + cfg_.anchorSize;
        if (endAnchor > n) endAnchor = n;
        for (size_t j = r.endSample + 1; j < endAnchor; ++j) {
            r.anchorsAfter.push_back(audio[j]);
        }
    }
}

// ── Severity classification ──────────────────────────────────
ClipSeverity ClipDetector::classifySeverity(size_t length) {
    if (length <= 4)  return ClipSeverity::Mild;
    if (length <= 16) return ClipSeverity::Moderate;
    if (length <= 64) return ClipSeverity::Severe;
    return ClipSeverity::Critical;
}

}  // namespace faurge
