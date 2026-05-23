#pragma once

namespace faurge {

class GainComputer {
public:
    GainComputer();

    void reset();

    float computeGainDb(float envelopeLinear, float thresholdDb,
                        float ratio, float kneeDb);

    float smoothGainDb(float targetGainDb, float attackMs,
                       float releaseMs, float holdMs, int sampleRate);

    bool isInHold() const { return holdTimer_ > 0; }

private:
    float smoothedGainDb_;
    int holdTimer_;
    float holdGainDb_;
    int holdSamples_;
};

}
