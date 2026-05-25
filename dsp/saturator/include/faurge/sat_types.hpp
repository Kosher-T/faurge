#pragma once

#include <cstddef>
#include <string>

namespace faurge {

enum class SatType : int {
    tube = 0,
    tape = 1,
    diode = 2,
    asymmetric = 3
};

struct SatConfig {
    float drive_db       = 0.0f;
    int   sat_type       = 0;
    float hpf_hz         = 20.0f;
    float lpf_hz         = 20000.0f;
    float mix            = 1.0f;
    int   oversampling   = 1;
    float output_trim_db = 0.0f;

    bool jsonOutput = false;
    bool verbose    = false;
};

struct SatResult {
    float processingTimeMs  = 0.0f;
    float inputPeakDb       = -120.0f;
    float outputPeakDb      = -120.0f;
    float inputRmsDb        = -120.0f;
    float outputRmsDb       = -120.0f;
    float avgHarmonicDb     = -120.0f;
    float dcOffset          = 0.0f;
    size_t framesProcessed  = 0;
    bool   success          = false;
    std::string errorMessage;
};

inline const char* satTypeToString(SatType t) {
    switch (t) {
        case SatType::tube:       return "tube";
        case SatType::tape:       return "tape";
        case SatType::diode:      return "diode";
        case SatType::asymmetric: return "asymmetric";
        default:                  return "unknown";
    }
}

} // namespace faurge
