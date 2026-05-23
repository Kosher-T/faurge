#include "faurge/gain_computer.hpp"

#include <algorithm>
#include <cmath>

namespace faurge {

GainComputer::GainComputer()
    : smoothedGainDb_(0.0f), holdTimer_(0), holdGainDb_(0.0f), holdSamples_(0) {}

void GainComputer::reset() {
    smoothedGainDb_ = 0.0f;
    holdTimer_ = 0;
    holdGainDb_ = 0.0f;
    holdSamples_ = 0;
}

float GainComputer::computeGainDb(float envelopeLinear, float thresholdDb,
                                   float ratio, float kneeDb) {
    const float eps = 1e-30f;
    float envDb = 20.0f * std::log10(std::max(envelopeLinear, eps));

    float overshoot = envDb - thresholdDb;

    if (overshoot <= -kneeDb * 0.5f) {
        return 0.0f;
    }

    float gr;
    if (overshoot >= kneeDb * 0.5f) {
        gr = -overshoot * (ratio - 1.0f) / ratio;
    } else {
        float x = overshoot / kneeDb + 0.5f;
        x = std::max(0.0f, std::min(1.0f, x));
        gr = -x * x * kneeDb * 0.5f * (ratio - 1.0f) / ratio;
    }

    return std::max(-120.0f, std::min(0.0f, gr));
}

float GainComputer::smoothGainDb(float targetGainDb, float attackMs,
                                  float releaseMs, float holdMs,
                                  int sampleRate) {
    if (holdMs > 0.0f && sampleRate > 0) {
        holdSamples_ = static_cast<int>(holdMs * static_cast<float>(sampleRate) * 0.001f);
    } else {
        holdSamples_ = 0;
    }

    if (targetGainDb < smoothedGainDb_ - 0.001f) {
        holdTimer_ = 0;
        float alpha = (attackMs <= 0.0f) ? 1.0f : (1.0f - std::exp(-2.2f / (attackMs * static_cast<float>(sampleRate) * 0.001f)));
        alpha = std::min(1.0f, alpha);
        smoothedGainDb_ += alpha * (targetGainDb - smoothedGainDb_);
        holdGainDb_ = smoothedGainDb_;
    } else if (targetGainDb > smoothedGainDb_ + 0.001f) {
        if (holdTimer_ > 0) {
            --holdTimer_;
            smoothedGainDb_ = holdGainDb_;
        } else {
            float alpha = (releaseMs <= 0.0f) ? 1.0f : (1.0f - std::exp(-2.2f / (releaseMs * static_cast<float>(sampleRate) * 0.001f)));
            alpha = std::min(1.0f, alpha);
            smoothedGainDb_ += alpha * (targetGainDb - smoothedGainDb_);
        }
    }

    if (holdSamples_ > 0 && holdTimer_ == 0 &&
        std::fabs(targetGainDb - smoothedGainDb_) < 0.001f) {
        holdTimer_ = holdSamples_;
        holdGainDb_ = smoothedGainDb_;
    }

    return smoothedGainDb_;
}

}
