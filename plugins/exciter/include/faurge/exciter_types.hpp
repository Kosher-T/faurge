#pragma once

#include <cstddef>
#include <string>

namespace faurge {

struct ExciterConfig {
    float highDriveDb     = 3.0f;
    float highMix         = 0.50f;
    float highCrossoverHz = 2000.0f;
    bool  highEnable      = true;

    float lowDriveDb      = 0.0f;
    float lowMix          = 0.35f;
    float lowCrossoverHz  = 200.0f;
    float lowSubLevel     = 0.50f;
    bool  lowEnable       = true;

    float masterVolume    = 1.0f;

    bool  jsonOutput      = false;
    bool  verbose         = false;
};

struct ExciterResult {
    float processingTimeMs   = 0.0f;
    float inputPeakDb        = -120.0f;
    float outputPeakDb       = -120.0f;
    float inputRmsDb         = -120.0f;
    float outputRmsDb        = -120.0f;
    float highBandEnergyDb   = -120.0f;
    float lowBandEnergyDb    = -120.0f;
    size_t framesProcessed   = 0;
    bool   success           = false;
    std::string errorMessage;
};

}
