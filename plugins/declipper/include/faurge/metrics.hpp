// ═══════════════════════════════════════════════════════════════
// Faurge Specialist — De-Clipper: Stage 5 — Metrics & Reporting
// ═══════════════════════════════════════════════════════════════
// Computes quality metrics and outputs JSON for the training
// pipeline, or a human-readable table to stderr.
// ═══════════════════════════════════════════════════════════════
#pragma once

#include "clip_region.hpp"
#include <string>
#include <vector>

namespace faurge {

class Metrics {
public:
    explicit Metrics(const DeclipConfig& config);

    // Compute per-region and aggregate metrics.
    // beforeAudio/afterAudio are the pre/post-declip buffers.
    std::vector<RegionMetric> compute(
        const float* beforeAudio,
        const float* afterAudio,
        size_t numSamples,
        const ClipReport& report) const;

    // THD+N estimation for a buffer (windowed, single-frequency assumption)
    static float estimateThdnDb(const float* audio, size_t numSamples,
                                int sampleRate);

    // Serialize DeclipResult to JSON string
    static std::string toJson(const DeclipResult& result);

    // Print human-readable summary table to stderr
    static void printSummary(const DeclipResult& result);

private:
    DeclipConfig cfg_;
};

}  // namespace faurge
