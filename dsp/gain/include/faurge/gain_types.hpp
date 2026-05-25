#pragma once

#include <cstddef>
#include <string>

namespace faurge {

struct GainConfig {
    float gain_db           = 0.0f;      // [-12, +12] dB
    float stereo_balance    = 0.0f;      // [-1, 1]

    bool jsonOutput = false;
    bool verbose    = false;
};

struct GainResult {
    float processingTimeMs       = 0.0f;
    float inputPeakDb            = -120.0f;
    float outputPeakDb           = -120.0f;
    float inputRmsDb             = -120.0f;
    float outputRmsDb            = -120.0f;
    float inputLufs              = -120.0f;
    float outputLufs             = -120.0f;
    float peakChangeDb           = 0.0f;
    float rmsChangeDb            = 0.0f;
    float appliedBalance         = 0.0f;
    bool  clipping               = false;
    size_t framesProcessed       = 0;
    bool   success               = false;
    std::string errorMessage;
};

} // namespace faurge
