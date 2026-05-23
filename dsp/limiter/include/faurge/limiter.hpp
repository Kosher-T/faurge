#pragma once

#include "lim_types.hpp"
#include "peak_predictor.hpp"
#include "oversampler.hpp"

#include <string>
#include <vector>

namespace faurge {

class Limiter {
public:
    explicit Limiter(const LimiterConfig& config = {});

    LimiterResult process(std::vector<float>& audio, int sampleRate);
    LimiterResult process(std::vector<float>& audio, int sampleRate,
                          const LimiterConfig& config);
    LimiterResult processFile(const std::string& inputPath,
                              const std::string& outputPath);

    const LimiterConfig& config() const { return cfg_; }

private:
    LimiterConfig cfg_;
    PeakPredictor predictor_;
    Oversampler oversampler_;
    float smoothedGrDb_ = 0.0f;

    static float dbToLinear(float db);
    static float linearToDb(float linear);
    static float peakDb(const float* audio, size_t n);
    static float rmsDb(const float* audio, size_t n);

    LimiterResult processImpl(std::vector<float>& audio, int sampleRate,
                              const LimiterConfig& config);
};

} // namespace faurge
