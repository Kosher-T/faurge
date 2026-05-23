#pragma once

#include <cstddef>
#include <string>

namespace faurge {

struct EsserConfig {
    float center_freq_hz  = 6000.0f;
    float threshold_db    = -30.0f;
    float ratio           = 5.0f;
    float bandwidth_hz    = 1500.0f;
    float attack_ms       = 2.0f;
    float release_ms      = 100.0f;

    bool jsonOutput = false;
    bool verbose    = false;
};

struct EsserResult {
    float processingTimeMs         = 0.0f;
    float inputPeakDb              = -120.0f;
    float outputPeakDb             = -120.0f;
    float inputRmsDb               = -120.0f;
    float outputRmsDb              = -120.0f;
    float maxGainReductionDb       = 0.0f;
    float avgActiveGainReductionDb = 0.0f;
    int   sibilantFrames           = 0;
    size_t framesProcessed         = 0;
    bool   success                 = false;
    std::string errorMessage;
};

} // namespace faurge
