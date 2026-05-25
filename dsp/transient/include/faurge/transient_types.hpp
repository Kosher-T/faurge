#pragma once

#include <cstddef>
#include <string>

namespace faurge {

struct TransientConfig {
    float attack_gain_db    = 6.0f;     // [-24, +24] dB
    float sustain_gain_db   = 0.0f;     // [-24, +24] dB
    float attack_time_ms    = 5.0f;     // [0.1, 50] ms
    float release_time_ms   = 100.0f;   // [10, 500] ms
    float sensitivity_db    = -30.0f;   // [-30, 0] dB
    float mix               = 1.0f;     // [0, 1]

    bool jsonOutput = false;
    bool verbose    = false;
};

struct TransientResult {
    float processingTimeMs       = 0.0f;
    float inputPeakDb            = -120.0f;
    float outputPeakDb           = -120.0f;
    float inputRmsDb             = -120.0f;
    float outputRmsDb            = -120.0f;
    float peakToRmsDb            = 0.0f;
    float avgAttackDb            = 0.0f;
    float avgSustainDb           = 0.0f;
    size_t framesProcessed       = 0;
    bool   success               = false;
    std::string errorMessage;
};

} // namespace faurge
