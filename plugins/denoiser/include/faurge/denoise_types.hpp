#pragma once

#include <string>
#include <vector>

namespace faurge {

struct DenoiseConfig {
    float  attenLimit      = 0.78f;
    std::string modelPath  = "";
    bool   jsonOutput      = false;
    bool   verbose         = false;
};

struct DenoiseResult {
    float  inputSnrEstDb    = 0.0f;
    float  outputSnrEstDb   = 0.0f;
    float  noiseFloorDbfs   = 0.0f;
    float  processingTimeMs = 0.0f;
    size_t framesProcessed  = 0;
    bool   success          = false;
    std::string errorMessage;
};

}  // namespace faurge
