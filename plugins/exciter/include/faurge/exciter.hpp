#pragma once

#include "exciter_types.hpp"
#include <string>
#include <vector>

namespace faurge {

class Exciter {
public:
    explicit Exciter(const ExciterConfig& config = {});

    ExciterResult process(std::vector<float>& audio, int sampleRate);
    ExciterResult processFile(const std::string& inputPath,
                              const std::string& outputPath);

    const ExciterConfig& config() const { return cfg_; }

private:
    ExciterConfig cfg_;

    static float dbToLinear(float db);
    static float peakDb(const float* audio, size_t n);
    static float rmsDb(const float* audio, size_t n);
    static float bandEnergyDb(const float* audio, size_t n);
};

}
