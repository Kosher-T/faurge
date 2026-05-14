#pragma once

#include "denoise_types.hpp"
#include <string>

namespace faurge {

class DenoiseMetrics {
public:
    static std::string toJson(const DenoiseResult& result);
    static void printSummary(const DenoiseResult& result);
    static float computeNoiseReductionRatioDb(const DenoiseResult& result);
};

}  // namespace faurge
