#include "faurge/noise_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace faurge {

static constexpr size_t kWindowSize = 1024;
static constexpr size_t kHistoryLen = 64;

float NoiseEstimator::computeRmsDb(const float* audio, size_t numSamples) const {
    if (numSamples == 0) return -120.0f;

    double sumSq = 0.0;
    for (size_t i = 0; i < numSamples; ++i) {
        sumSq += static_cast<double>(audio[i]) * audio[i];
    }
    double rms = std::sqrt(sumSq / numSamples);
    if (rms < 1e-20) return -120.0f;

    return 20.0f * std::log10(static_cast<float>(rms));
}

float NoiseEstimator::estimateNoiseFloorDb(const float* audio,
                                            size_t numSamples,
                                            int sampleRate) {
    if (numSamples == 0 || sampleRate <= 0) return -120.0f;

    size_t hop = kWindowSize / 2;
    noiseFloorHistory_.resize(kHistoryLen, -120.0f);
    historyPos_ = 0;

    std::vector<float> minPerWindow;

    for (size_t start = 0; start + kWindowSize <= numSamples; start += hop) {
        float rmsDb = computeRmsDb(audio + start, kWindowSize);
        noiseFloorHistory_[historyPos_ % kHistoryLen] = rmsDb;
        ++historyPos_;

        if (historyPos_ >= kHistoryLen) {
            float minVal = *std::min_element(
                noiseFloorHistory_.begin(),
                noiseFloorHistory_.end());
            minPerWindow.push_back(minVal);
        }
    }

    if (minPerWindow.empty()) {
        return computeRmsDb(audio, numSamples);
    }

    float floor = std::accumulate(minPerWindow.begin(), minPerWindow.end(), 0.0f);
    floor /= static_cast<float>(minPerWindow.size());

    return floor;
}

float NoiseEstimator::estimateSnrDb(const float* audio, size_t numSamples,
                                     int sampleRate) {
    if (numSamples == 0 || sampleRate <= 0) return 0.0f;

    float noiseFloor = estimateNoiseFloorDb(audio, numSamples, sampleRate);
    float signalRmsDb = computeRmsDb(audio, numSamples);

    float snr = signalRmsDb - noiseFloor;
    if (snr < 0.0f) snr = 0.0f;

    return snr;
}

}  // namespace faurge
