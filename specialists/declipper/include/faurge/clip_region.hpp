// ═══════════════════════════════════════════════════════════════
// Faurge Specialist — De-Clipper: Core Data Structures
// ═══════════════════════════════════════════════════════════════
// Defines the shared types used across all de-clipper stages.
// These structures are intentionally plain-old-data to make the
// reconstruction algorithm easy to extract for portability.
// ═══════════════════════════════════════════════════════════════
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace faurge {

// ── Severity Classification ──────────────────────────────────
// How badly the clip region is damaged.  Determines which
// reconstruction strategy the Reconstructor selects.
enum class ClipSeverity {
    Mild,      // 1–4 consecutive clipped samples
    Moderate,  // 5–16 consecutive clipped samples
    Severe,    // 17–64 consecutive clipped samples
    Critical   // >64 consecutive clipped samples (sustained clip)
};

inline const char* severityToString(ClipSeverity s) {
    switch (s) {
        case ClipSeverity::Mild:     return "mild";
        case ClipSeverity::Moderate: return "moderate";
        case ClipSeverity::Severe:   return "severe";
        case ClipSeverity::Critical: return "critical";
    }
    return "unknown";
}

// ── Clip Polarity ────────────────────────────────────────────
// Whether the waveform was chopped at the positive or negative
// rail.  Required so the reconstructor draws the peak in the
// correct direction.
enum class ClipPolarity {
    Positive,  // Signal flatlined at +threshold
    Negative   // Signal flatlined at -threshold
};

// ── A Single Clipped Region ──────────────────────────────────
// Represents one contiguous run of clipped samples.
struct ClipRegion {
    size_t startSample  = 0;   // First clipped sample index (inclusive)
    size_t endSample    = 0;   // Last clipped sample index (inclusive)
    ClipPolarity polarity = ClipPolarity::Positive;
    ClipSeverity severity = ClipSeverity::Mild;

    // Boundary context — filled by the detector
    std::vector<float> anchorsBefore;  // N good samples immediately before clip
    std::vector<float> anchorsAfter;   // N good samples immediately after clip

    // Estimated true peak that was lost (filled by the reconstructor)
    float estimatedPeakAmplitude = 0.0f;

    // Convenience
    size_t length() const { return endSample - startSample + 1; }
};

// ── Aggregate Report ─────────────────────────────────────────
// Summary output from the detection stage.
struct ClipReport {
    std::vector<ClipRegion> regions;
    size_t totalClippedSamples = 0;
    size_t totalSamples        = 0;
    float  percentClipped      = 0.0f;
};

// ── Per-Region Metrics (post-reconstruction) ─────────────────
struct RegionMetric {
    size_t regionIndex         = 0;
    size_t lengthSamples       = 0;
    ClipSeverity severity      = ClipSeverity::Mild;
    float  estimatedOvershootDb = 0.0f;  // How far above threshold the peak was
    float  reconstructionSnrDb  = 0.0f;  // Only meaningful if ground truth exists
};

// ── Full Declip Result ───────────────────────────────────────
struct DeclipResult {
    ClipReport report;
    std::vector<RegionMetric> regionMetrics;

    // Aggregate quality metrics
    float beforeThdnDb = 0.0f;  // THD+N estimate before declipping
    float afterThdnDb  = 0.0f;  // THD+N estimate after declipping
    float processingTimeMs = 0.0f;

    bool  success = false;
    std::string errorMessage;
};

// ── Configuration ────────────────────────────────────────────
// Every tunable knob in one struct.  Sensible defaults set here.
struct DeclipConfig {
    // Detection
    float  clipThreshold    = 0.9999f;  // Absolute sample value (0–1 range)
    int    mergeGap         = 3;        // Merge clips separated by ≤ this many samples
    int    minClipLength    = 2;        // Ignore clips shorter than this
    int    anchorSize       = 4;        // Boundary context samples on each side
    bool   detectSoftClip   = true;     // Also look for derivative discontinuities
    float  softClipDerivThr = 0.85f;    // Second-derivative threshold for soft-clip

    // Reconstruction
    int    hermiteMaxLen    = 16;       // Use Cubic Hermite for clips ≤ this
    int    akimaMaxLen      = 64;       // Use Akima for clips ≤ this (above = AR)
    int    arModelOrder     = 14;       // Burg AR model order for long clips
    float  peakOvershoot    = 1.15f;    // Max estimated peak = threshold × this

    // Post-processing
    int    crossfadeWidth   = 8;        // Raised-cosine crossfade samples at boundaries
    float  dcBlockFreqHz    = 10.0f;    // High-pass cutoff for DC removal
    bool   enableAntiAlias  = true;     // Low-pass reconstructed regions

    // Output
    bool   jsonOutput       = false;    // Dump metrics as JSON
    bool   verbose          = false;    // Per-region logging to stderr
};

}  // namespace faurge
