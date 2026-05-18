#pragma once

#include <cstddef>
#include <vector>

namespace faurge {

class LowBand {
public:
    LowBand();

    void process(const float* input, float* output,
                 size_t numSamples, int sampleRate,
                 float driveDb, float subLevel);

private:
    std::vector<float> rectBuf_;
    std::vector<float> lpBuf_;

    struct BiquadState {
        float x1 = 0.0f, x2 = 0.0f, y1 = 0.0f, y2 = 0.0f;
    };
    BiquadState lp1_, lp2_;

    float prevLpCoeff_ = 0.0f;
    int prevSampleRate_ = 0;

    void initLpf(int sampleRate);
    void applyLpf(const float* in, float* out, size_t n);
};

}
