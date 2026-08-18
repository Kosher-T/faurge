#pragma once

#include "declipper_types.hpp"

#include <string>
#include <vector>

namespace faurge {

class Declipper {
public:
    explicit Declipper(const DeclipConfig& config = {});
    DeclipResult process(std::vector<float>& audio, int sampleRate);

    const DeclipConfig& config() const { return cfg_; }

private:
    DeclipConfig cfg_;
};

void clip_audio_inplace(std::vector<float>& audio, float gainDb, bool softClip);

} // namespace faurge
