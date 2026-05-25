#pragma once

#include "transient_types.hpp"
#include "transient_splitter.hpp"

#include <string>
#include <vector>

namespace faurge {

class TransientShaper {
public:
    explicit TransientShaper(const TransientConfig& config = {});

    TransientResult process(std::vector<float>& audio, int sampleRate);
    TransientResult process(std::vector<float>& audio, int sampleRate,
                            const TransientConfig& config);
    TransientResult processFile(const std::string& inputPath,
                                const std::string& outputPath);

    const TransientConfig& config() const { return cfg_; }

private:
    TransientConfig cfg_;
    TransientSplitter splitter_;

    static float dbToLinear(float db);
    static float linearToDb(float linear);
    static float peakDb(const float* audio, size_t n);
    static float rmsDb(const float* audio, size_t n);

    TransientResult processImpl(std::vector<float>& audio, int sampleRate,
                                const TransientConfig& config);
};

} // namespace faurge
