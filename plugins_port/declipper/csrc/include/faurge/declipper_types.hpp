#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace faurge {

enum class ClipSeverity {
    Mild     = 0,
    Moderate = 1,
    Severe   = 2,
    Critical = 3
};

enum class ClipPolarity {
    Positive = 0,
    Negative = 1
};

struct ClipRegion {
    size_t startSample = 0;
    size_t endSample   = 0;
    ClipPolarity polarity = ClipPolarity::Positive;
    ClipSeverity severity  = ClipSeverity::Mild;

    std::vector<float> anchorsBefore;
    std::vector<float> anchorsAfter;

    float estimatedPeakAmplitude = 0.0f;

    size_t length() const {
        if (endSample < startSample) return 0;
        return endSample - startSample + 1;
    }
};

struct ClipReport {
    std::vector<ClipRegion> regions;
    size_t totalSamples        = 0;
    size_t totalClippedSamples = 0;
    float  percentClipped      = 0.0f;
};

struct RegionMetric {
    size_t  regionIndex          = 0;
    size_t  lengthSamples        = 0;
    ClipSeverity severity        = ClipSeverity::Mild;
    float   estimatedOvershootDb = -120.0f;
    float   reconstructionSnrDb = 0.0f;
};

struct DeclipResult {
    bool   success        = false;
    float  processingTimeMs = 0.0f;

    float  beforeThdnDb   = -120.0f;
    float  afterThdnDb    = -120.0f;

    ClipReport                  report;
    std::vector<RegionMetric>   regionMetrics;

    std::string errorMessage;
};

struct DeclipConfig {
    // Detection
    float  clipThreshold    = 0.9999f;
    int    minClipLength    = 2;
    int    mergeGap         = 3;
    int    anchorSize       = 20;
    bool   detectSoftClip   = true;
    float  softClipDerivThr = 0.5f;

    // Reconstruction strategy thresholds
    int    hermiteMaxLen    = 16;
    int    akimaMaxLen      = 64;
    int    arModelOrder     = 14;

    // Peak estimation
    float  peakOvershoot    = 1.15f;

    // Post-processing
    int    crossfadeWidth   = 8;
    bool   enableAntiAlias  = true;
    float  dcBlockFreqHz    = 10.0f;

    // Reporting
    bool   jsonOutput       = false;
    bool   verbose          = false;
};

inline const char* severityToString(ClipSeverity s) {
    switch (s) {
        case ClipSeverity::Mild:     return "Mild";
        case ClipSeverity::Moderate: return "Moderate";
        case ClipSeverity::Severe:   return "Severe";
        case ClipSeverity::Critical: return "Critical";
    }
    return "Unknown";
}

} // namespace faurge
