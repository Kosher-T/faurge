#include "faurge/transient_splitter.hpp"

#include <algorithm>
#include <cmath>

namespace faurge {

TransientSplitter::TransientSplitter() : envelope_(0.0f) {}

void TransientSplitter::reset() {
    envelope_ = 0.0f;
}

float TransientSplitter::rcAlpha(float tauMs, int sampleRate) {
    if (tauMs <= 0.0f || sampleRate <= 0) return 1.0f;
    float samples = tauMs * static_cast<float>(sampleRate) * 0.001f;
    return 1.0f - std::exp(-2.2f / samples);
}

void TransientSplitter::processFactors(const float* audio, size_t n,
                                        float attackTimeMs, float releaseTimeMs,
                                        float sensitivityDb, int sampleRate,
                                        float* factors) {
    if (n == 0) return;

    float sensLin = std::pow(10.0f, sensitivityDb / 20.0f);

    for (size_t i = 0; i < n; ++i) {
        float absX = std::fabs(audio[i]);

        if (absX > envelope_) {
            float alpha = rcAlpha(attackTimeMs, sampleRate);
            envelope_ += alpha * (absX - envelope_);
        } else {
            float alpha = rcAlpha(releaseTimeMs, sampleRate);
            envelope_ += alpha * (absX - envelope_);
        }

        if (std::max(absX, envelope_) < sensLin) {
            factors[i] = 0.0f;
            continue;
        }

        float denom = std::max(absX, envelope_);
        if (denom > 1e-30f) {
            float diff = absX - envelope_;
            if (diff > 0.0f) {
                factors[i] = diff / denom;
            } else {
                factors[i] = -(-diff) / denom;
            }
        } else {
            factors[i] = 0.0f;
        }
    }
}

void TransientSplitter::applyGain(const float* input, float* output, size_t n,
                                   const float* factors,
                                   float attackLin, float sustainLin, float mix) {
    for (size_t i = 0; i < n; ++i) {
        float f = factors[i];
        float gain;
        if (f > 0.0f) {
            gain = 1.0f + (attackLin - 1.0f) * f;
        } else if (f < 0.0f) {
            gain = 1.0f + (sustainLin - 1.0f) * (-f);
        } else {
            gain = 1.0f;
        }
        float wet = input[i] * gain;
        output[i] = wet * mix + input[i] * (1.0f - mix);
    }
}

} // namespace faurge
