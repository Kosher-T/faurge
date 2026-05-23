#pragma once

#include <cstddef>
#include <string>

namespace faurge {

enum class FilterType : int {
    peak = 0,
    low_shelf,
    high_shelf,
    highpass,
    lowpass,
    bandpass,
    notch
};

constexpr int NUM_EQ_BANDS = 31;

struct FilterBand {
    float freq_hz         = 1000.0f;
    float gain_db         = 0.0f;
    float q               = 1.0f;
    FilterType filter_type = FilterType::peak;
    float stereo_skew_db  = 0.0f;
    float dynamic_depth   = 0.0f;
};

struct EqConfig {
    FilterBand bands[NUM_EQ_BANDS];
    bool  jsonOutput = false;
    bool  verbose    = false;
};

struct EqResult {
    float processingTimeMs   = 0.0f;
    float inputPeakDb        = -120.0f;
    float outputPeakDb       = -120.0f;
    float inputRmsDb         = -120.0f;
    float outputRmsDb        = -120.0f;
    size_t framesProcessed   = 0;
    bool   success           = false;
    std::string errorMessage;
};

inline const char* filterTypeToString(FilterType t) {
    switch (t) {
        case FilterType::peak:       return "peak";
        case FilterType::low_shelf:  return "low_shelf";
        case FilterType::high_shelf: return "high_shelf";
        case FilterType::highpass:   return "highpass";
        case FilterType::lowpass:    return "lowpass";
        case FilterType::bandpass:   return "bandpass";
        case FilterType::notch:      return "notch";
        default:                     return "unknown";
    }
}

}
