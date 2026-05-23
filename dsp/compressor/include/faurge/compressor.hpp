#pragma once

#include "comp_types.hpp"
#include "envelope.hpp"
#include "gain_computer.hpp"

#include <string>
#include <vector>

namespace faurge {

class Compressor {
public:
    explicit Compressor(const CompConfig& config = {});

    CompResult process(std::vector<float>& audio, int sampleRate);
    CompResult process(std::vector<float>& audio, int sampleRate, const CompConfig& config);
    CompResult processFile(const std::string& inputPath,
                           const std::string& outputPath);

    const CompConfig& config() const { return cfg_; }

private:
    CompConfig cfg_;
    EnvelopeFollower envFollower_;
    GainComputer gainComputer_;

    static float dbToLinear(float db);
    static float linearToDb(float linear);
    static float peakDb(const float* audio, size_t n);
    static float rmsDb(const float* audio, size_t n);

    CompResult processImpl(std::vector<float>& audio, int sampleRate,
                           const CompConfig& config);
};

}
