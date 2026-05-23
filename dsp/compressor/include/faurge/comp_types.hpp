#pragma once

#include <cstddef>
#include <string>

namespace faurge {

enum class DetectorType : int {
    RMS = 0,
    peak,
    feed_forward,
    feed_back
};

struct CompConfig {
    float threshold_db       = -24.0f;
    float ratio              = 4.0f;
    float attack_ms          = 5.0f;
    float release_ms         = 150.0f;
    float knee_db            = 6.0f;
    float lookahead_ms       = 0.0f;
    float hold_ms            = 0.0f;
    float wet_dry_mix        = 1.0f;
    float stereo_link        = 1.0f;
    float sidechain_hp_hz    = 20.0f;
    float sidechain_lp_hz    = 20000.0f;
    float saturate_drive_db  = 0.0f;
    float output_trim_db     = 0.0f;
    DetectorType detector_type = DetectorType::RMS;
    bool jsonOutput = false;
    bool verbose    = false;
};

struct CompResult {
    float processingTimeMs     = 0.0f;
    float inputPeakDb          = -120.0f;
    float outputPeakDb         = -120.0f;
    float inputRmsDb           = -120.0f;
    float outputRmsDb          = -120.0f;
    float gainReductionDb      = 0.0f;
    float avgGainReductionDb   = 0.0f;
    size_t framesProcessed     = 0;
    bool   success             = false;
    std::string errorMessage;
};

inline const char* detectorTypeToString(DetectorType t) {
    switch (t) {
        case DetectorType::RMS:           return "RMS";
        case DetectorType::peak:          return "peak";
        case DetectorType::feed_forward:  return "feed_forward";
        case DetectorType::feed_back:     return "feed_back";
        default:                          return "unknown";
    }
}

}
