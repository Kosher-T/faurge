#pragma once

#include "sat_types.hpp"
#include "biquad.hpp"

#include <string>
#include <vector>

namespace faurge {

class Saturator {
public:
    explicit Saturator(const SatConfig& config = {});

    SatResult process(std::vector<float>& audio, int sampleRate);
    SatResult process(std::vector<float>& audio, int sampleRate,
                      const SatConfig& config);
    SatResult processFile(const std::string& inputPath,
                          const std::string& outputPath);

    const SatConfig& config() const { return cfg_; }

private:
    SatConfig cfg_;

    static float dbToLinear(float db);
    static float linearToDb(float linear);
    static float peakDb(const float* audio, size_t n);
    static float rmsDb(const float* audio, size_t n);

    void processWetPath(float* buf, size_t n, int sampleRate,
                        float& harmSum);
    void processOversampled(float* buf, size_t n, int sampleRate);
    void applyWaveshaper(float* buf, size_t n, float& harmSum);
    void applyBiquads(float* buf, size_t n, const BiquadCoeffs& c1,
                      const BiquadCoeffs& c2);

    SatResult processImpl(std::vector<float>& audio, int sampleRate,
                          const SatConfig& config);
};

} // namespace faurge
