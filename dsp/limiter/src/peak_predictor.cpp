#include "faurge/peak_predictor.hpp"

#include <algorithm>
#include <cmath>

namespace faurge {

PeakPredictor::PeakPredictor() {}

void PeakPredictor::reset() {
    writeIdx_ = 0;
    if (!buf_.empty()) {
        std::fill(buf_.begin(), buf_.end(), 0.0f);
    }
}

void PeakPredictor::configure(int lookaheadSamples) {
    size_ = lookaheadSamples;
    buf_.resize(static_cast<size_t>(size_), 0.0f);
    writeIdx_ = 0;
}

void PeakPredictor::processSample(float input, float& delayedOut,
                                  float& predictedPeakLin) {
    if (size_ <= 0) {
        delayedOut = input;
        predictedPeakLin = std::fabs(input);
        return;
    }

    buf_[writeIdx_] = input;

    int readIdx = (writeIdx_ + 1) % size_;
    delayedOut = buf_[readIdx];

    float peak = 0.0f;
    for (int i = 0; i < size_; ++i) {
        float absVal = std::fabs(buf_[i]);
        if (absVal > peak) peak = absVal;
    }
    predictedPeakLin = peak;

    writeIdx_ = (writeIdx_ + 1) % size_;
}

} // namespace faurge
