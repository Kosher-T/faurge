#pragma once

#include <cstddef>
#include <vector>

namespace faurge {

class HighBand {
public:
    HighBand();

    void process(const float* input, float* output,
                 size_t numSamples, int sampleRate, float driveDb);

private:
    std::vector<float> oversampleBuf_;
    std::vector<float> filterBuf_;

    struct BiquadState { float x1 = 0.0f, x2 = 0.0f, y1 = 0.0f, y2 = 0.0f; };
    BiquadState aaState_;

    void initAntiAlias(int sampleRate);
    void applyAntiAlias(float* buf, size_t n);
};

}
