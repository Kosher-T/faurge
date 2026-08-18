#pragma once

#include "declipper_types.hpp"

#include <string>
#include <vector>

namespace faurge {

class Metrics {
public:
    explicit Metrics(const DeclipConfig& config);

    std::vector<RegionMetric> compute(
        const float* beforeAudio,
        const float* afterAudio,
        size_t numSamples,
        const ClipReport& report) const;

    static float estimateThdnDb(const float* audio, size_t numSamples, int sampleRate);
    static std::string toJson(const DeclipResult& result);
    static void printSummary(const DeclipResult& result);

private:
    DeclipConfig cfg_;
};

} // namespace faurge
