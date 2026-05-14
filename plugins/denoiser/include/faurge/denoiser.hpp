#pragma once

#include "denoise_types.hpp"
#include <memory>
#include <string>
#include <vector>

namespace faurge {

class Denoiser {
public:
    explicit Denoiser(const DenoiseConfig& config = {});
    ~Denoiser();

    DenoiseResult process(std::vector<float>& audio, int sampleRate);
    DenoiseResult processFile(const std::string& inputPath,
                              const std::string& outputPath);

    const DenoiseConfig& config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace faurge
