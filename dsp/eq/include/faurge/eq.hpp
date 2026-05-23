#pragma once

#include "eq_types.hpp"
#include "biquad.hpp"

#include <string>
#include <vector>

namespace faurge {

class Equalizer {
public:
    explicit Equalizer(const EqConfig& config = {});

    EqResult process(std::vector<float>& audio, int sampleRate);
    EqResult process(std::vector<float>& audio, int sampleRate, const EqConfig& config);
    EqResult process(std::vector<float>& audio, int sampleRate,
                     const EqConfig& config, int channel);
    EqResult processFile(const std::string& inputPath,
                         const std::string& outputPath);

    const EqConfig& config() const { return cfg_; }

private:
    EqConfig cfg_;
    Biquad bands_[NUM_EQ_BANDS];

    static float dbToLinear(float db);
    static float peakDb(const float* audio, size_t n);
    static float rmsDb(const float* audio, size_t n);
    EqResult processImpl(std::vector<float>& audio, int sampleRate,
                         const EqConfig& config, int channel = -1);
};

}
