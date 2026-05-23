#pragma once

#include <cstddef>
#include <string>

namespace faurge {

enum class ClipMode : int {
    hard = 0,
    soft = 1
};

struct LimiterConfig {
    float ceiling_db      = -1.0f;
    float release_ms      = 100.0f;
    float lookahead_ms    = 5.0f;
    ClipMode clip_mode    = ClipMode::soft;
    float stereo_link     = 1.0f;
    int   oversampling    = 1;

    bool jsonOutput = false;
    bool verbose    = false;
};

struct LimiterResult {
    float processingTimeMs       = 0.0f;
    float inputPeakDb            = -120.0f;
    float outputPeakDb           = -120.0f;
    float inputRmsDb             = -120.0f;
    float outputRmsDb            = -120.0f;
    float maxGainReductionDb     = 0.0f;
    float avgGainReductionDb     = 0.0f;
    int   clippedSamples         = 0;
    size_t framesProcessed       = 0;
    bool   success               = false;
    std::string errorMessage;
};

inline const char* clipModeToString(ClipMode m) {
    switch (m) {
        case ClipMode::hard: return "hard";
        case ClipMode::soft: return "soft";
        default:             return "unknown";
    }
}

} // namespace faurge
