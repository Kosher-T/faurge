#pragma once

#include <cstddef>

namespace faurge {

class TransientSplitter {
public:
    TransientSplitter();

    void reset();

    void processFactors(const float* audio, size_t n,
                        float attackTimeMs, float releaseTimeMs,
                        float sensitivityDb, int sampleRate,
                        float* factors);

    static void applyGain(const float* input, float* output, size_t n,
                          const float* factors,
                          float attackLin, float sustainLin, float mix);

    float currentEnvelope() const { return envelope_; }

private:
    float envelope_;

    static float rcAlpha(float tauMs, int sampleRate);
};

} // namespace faurge
