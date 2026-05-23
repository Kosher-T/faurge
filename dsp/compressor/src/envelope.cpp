#include "faurge/envelope.hpp"

#include <algorithm>
#include <cmath>

namespace faurge {

static constexpr float PI = 3.14159265358979f;

EnvelopeFollower::EnvelopeFollower() : envelope_(0.0f), sampleRate_(0) {}

void EnvelopeFollower::reset() {
    envelope_ = 0.0f;
}

float EnvelopeFollower::rcAlpha(float tauMs, int sampleRate) {
    if (tauMs <= 0.0f || sampleRate <= 0) return 1.0f;
    float samples = tauMs * static_cast<float>(sampleRate) * 0.001f;
    return 1.0f - std::exp(-2.2f / samples);
}

float EnvelopeFollower::processSample(float input, DetectorType type,
                                      float attackMs, float releaseMs,
                                      int sampleRate) {
    sampleRate_ = static_cast<float>(sampleRate);

    float detected;

    switch (type) {
        case DetectorType::RMS:
            detected = input * input;
            break;
        case DetectorType::peak:
            detected = std::fabs(input);
            break;
        case DetectorType::feed_forward:
            detected = std::fabs(input);
            break;
        case DetectorType::feed_back:
            detected = std::fabs(input);
            break;
        default:
            detected = std::fabs(input);
            break;
    }

    float alpha;
    if (detected > envelope_) {
        alpha = rcAlpha(attackMs, sampleRate);
    } else {
        alpha = rcAlpha(releaseMs, sampleRate);
    }

    envelope_ += alpha * (detected - envelope_);

    if (type == DetectorType::RMS) {
        return std::sqrt(envelope_);
    }
    return envelope_;
}

void EnvelopeFollower::processBlock(const float* audio, float* envelope,
                                    size_t n, DetectorType type,
                                    float attackMs, float releaseMs,
                                    int sampleRate) {
    for (size_t i = 0; i < n; ++i) {
        envelope[i] = processSample(audio[i], type, attackMs, releaseMs, sampleRate);
    }
}

}
