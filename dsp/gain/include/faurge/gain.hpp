#pragma once

#include "gain_types.hpp"

#include <string>
#include <vector>

namespace faurge {

class Gain {
public:
    explicit Gain(const GainConfig& config = {});

    GainResult process(std::vector<float>& audio, int sampleRate, int channels);
    GainResult process(std::vector<float>& audio, int sampleRate, int channels,
                       const GainConfig& config);
    GainResult processFile(const std::string& inputPath,
                           const std::string& outputPath);

    const GainConfig& config() const { return cfg_; }

private:
    GainConfig cfg_;

    static float dbToLinear(float db);
    static float linearToDb(float linear);
    static float peakDb(const float* audio, size_t n);
    static float rmsDb(const float* audio, size_t n);

    static float measureLufs(const float* audio, size_t n,
                             int sampleRate, int channels);

    GainResult processImpl(std::vector<float>& audio, int sampleRate,
                           int channels, const GainConfig& config);
};

} // namespace faurge
