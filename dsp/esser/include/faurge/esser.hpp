#pragma once

#include "esser_types.hpp"
#include "sibilance_detector.hpp"

#include <string>
#include <vector>

namespace faurge {

class Esser {
public:
    explicit Esser(const EsserConfig& config = {});

    EsserResult process(std::vector<float>& audio, int sampleRate);
    EsserResult process(std::vector<float>& audio, int sampleRate,
                        const EsserConfig& config);
    EsserResult processFile(const std::string& inputPath,
                            const std::string& outputPath);

    const EsserConfig& config() const { return cfg_; }

private:
    EsserConfig cfg_;
    SibilanceDetector detector_;

    float envelope_       = 0.0f;
    float smoothedGainDb_ = 0.0f;
    int    sampleRate_    = 0;

    static float dbToLinear(float db);
    static float linearToDb(float linear);
    static float peakDb(const float* audio, size_t n);
    static float rmsDb(const float* audio, size_t n);

    EsserResult processImpl(std::vector<float>& audio, int sampleRate,
                            const EsserConfig& config);
};

} // namespace faurge
