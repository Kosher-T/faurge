#pragma once

#include <cstddef>
#include <vector>

namespace faurge {

class PeakPredictor {
public:
    PeakPredictor();
    void reset();
    void configure(int lookaheadSamples);

    void processSample(float input, float& delayedOut, float& predictedPeakLin);

private:
    std::vector<float> buf_;
    int writeIdx_ = 0;
    int size_ = 0;
};

} // namespace faurge
