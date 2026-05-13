// ═══════════════════════════════════════════════════════════════
// Faurge Specialist — De-Clipper: Metrics & Reporting
// ═══════════════════════════════════════════════════════════════
#include "faurge/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace faurge {

Metrics::Metrics(const DeclipConfig& config) : cfg_(config) {}

// ── Compute per-region metrics ───────────────────────────────
std::vector<RegionMetric> Metrics::compute(
        const float* beforeAudio,
        const float* afterAudio,
        size_t numSamples,
        const ClipReport& report) const {
    std::vector<RegionMetric> metrics;
    metrics.reserve(report.regions.size());

    for (size_t i = 0; i < report.regions.size(); ++i) {
        const auto& r = report.regions[i];
        RegionMetric m;
        m.regionIndex    = i;
        m.lengthSamples  = r.length();
        m.severity       = r.severity;

        // Estimated overshoot in dB
        if (r.estimatedPeakAmplitude > 0.0f) {
            m.estimatedOvershootDb = 20.0f * std::log10(
                r.estimatedPeakAmplitude / 1.0f);  // dB above 0 dBFS
        }

        // Reconstruction SNR (if we have before/after for the region)
        float sigPower = 0.0f;
        float noisePower = 0.0f;
        for (size_t j = r.startSample; j <= r.endSample && j < numSamples; ++j) {
            float sig   = afterAudio[j];
            float noise = afterAudio[j] - beforeAudio[j];
            sigPower   += sig * sig;
            noisePower += noise * noise;
        }
        if (noisePower > 1e-30f) {
            m.reconstructionSnrDb = 10.0f * std::log10(sigPower / noisePower);
        }

        metrics.push_back(m);
    }

    return metrics;
}

// ── THD+N estimation ─────────────────────────────────────────
// Simplified THD+N: measures the ratio of total harmonic
// distortion + noise to the fundamental.  Uses a basic DFT
// approach on a windowed segment.
//
// This is an approximation — not a lab-grade measurement — but
// sufficient for comparing before/after quality.
float Metrics::estimateThdnDb(const float* audio, size_t numSamples,
                              int /* sampleRate */) {
    if (numSamples < 64) return 0.0f;

    // Use a window up to 4096 samples from the middle of the buffer
    size_t fftSize = 1;
    while (fftSize * 2 <= numSamples && fftSize < 4096) fftSize *= 2;

    size_t offset = (numSamples - fftSize) / 2;

    // Apply Hann window and compute magnitude spectrum via DFT
    // (We do a real DFT manually since we have no external FFT lib)
    std::vector<float> windowed(fftSize);
    for (size_t i = 0; i < fftSize; ++i) {
        float w = 0.5f * (1.0f - std::cos(2.0f * 3.14159265f * i / fftSize));
        windowed[i] = audio[offset + i] * w;
    }

    // Find the bin with maximum energy (fundamental)
    std::vector<float> mag(fftSize / 2, 0.0f);
    for (size_t k = 1; k < fftSize / 2; ++k) {
        float re = 0.0f, im = 0.0f;
        for (size_t n = 0; n < fftSize; ++n) {
            float angle = 2.0f * 3.14159265f * k * n / fftSize;
            re += windowed[n] * std::cos(angle);
            im -= windowed[n] * std::sin(angle);
        }
        mag[k] = re * re + im * im;
    }

    // Find fundamental bin
    size_t fundBin = 1;
    float fundMag = mag[1];
    for (size_t k = 2; k < fftSize / 2; ++k) {
        if (mag[k] > fundMag) {
            fundMag = mag[k];
            fundBin = k;
        }
    }

    // Sum total power and subtract fundamental (±2 bins)
    float totalPower = 0.0f;
    float fundPower  = 0.0f;
    for (size_t k = 1; k < fftSize / 2; ++k) {
        totalPower += mag[k];
        if (k >= fundBin - 2 && k <= fundBin + 2) {
            fundPower += mag[k];
        }
    }

    float thdnPower = totalPower - fundPower;
    if (fundPower < 1e-30f) return -120.0f;

    return 10.0f * std::log10(thdnPower / fundPower);
}

// ── JSON serialisation ───────────────────────────────────────
std::string Metrics::toJson(const DeclipResult& result) {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"success\": " << (result.success ? "true" : "false") << ",\n";
    ss << "  \"processing_time_ms\": " << result.processingTimeMs << ",\n";
    ss << "  \"before_thdn_db\": " << result.beforeThdnDb << ",\n";
    ss << "  \"after_thdn_db\": " << result.afterThdnDb << ",\n";
    ss << "  \"clip_report\": {\n";
    ss << "    \"total_samples\": " << result.report.totalSamples << ",\n";
    ss << "    \"total_clipped_samples\": " << result.report.totalClippedSamples << ",\n";
    ss << "    \"percent_clipped\": " << result.report.percentClipped << ",\n";
    ss << "    \"num_regions\": " << result.report.regions.size() << "\n";
    ss << "  },\n";
    ss << "  \"regions\": [\n";

    for (size_t i = 0; i < result.regionMetrics.size(); ++i) {
        const auto& m = result.regionMetrics[i];
        ss << "    {\n";
        ss << "      \"index\": " << m.regionIndex << ",\n";
        ss << "      \"length_samples\": " << m.lengthSamples << ",\n";
        ss << "      \"severity\": \"" << severityToString(m.severity) << "\",\n";
        ss << "      \"estimated_overshoot_db\": " << m.estimatedOvershootDb << ",\n";
        ss << "      \"reconstruction_snr_db\": " << m.reconstructionSnrDb << "\n";
        ss << "    }";
        if (i + 1 < result.regionMetrics.size()) ss << ",";
        ss << "\n";
    }

    ss << "  ]\n";
    ss << "}\n";

    return ss.str();
}

// ── Human-readable summary ───────────────────────────────────
void Metrics::printSummary(const DeclipResult& result) {
    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  FAURGE DE-CLIPPER — PROCESSING REPORT      ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Status:           %-25s ║\n",
            result.success ? "SUCCESS" : "FAILED");
    fprintf(stderr, "║  Processing time:  %-21.2f ms ║\n",
            result.processingTimeMs);
    fprintf(stderr, "║  Total samples:    %-25zu ║\n",
            result.report.totalSamples);
    fprintf(stderr, "║  Clipped samples:  %-25zu ║\n",
            result.report.totalClippedSamples);
    fprintf(stderr, "║  Percent clipped:  %-22.2f %% ║\n",
            result.report.percentClipped);
    fprintf(stderr, "║  Clip regions:     %-25zu ║\n",
            result.report.regions.size());
    fprintf(stderr, "║  THD+N before:     %-21.1f dB ║\n",
            result.beforeThdnDb);
    fprintf(stderr, "║  THD+N after:      %-21.1f dB ║\n",
            result.afterThdnDb);
    fprintf(stderr, "╠══════════════════════════════════════════════╣\n");

    if (!result.regionMetrics.empty()) {
        fprintf(stderr, "║  #    Length  Severity    Overshoot   SNR    ║\n");
        fprintf(stderr, "║  ──── ─────── ─────────── ─────────── ────── ║\n");
        for (const auto& m : result.regionMetrics) {
            fprintf(stderr, "║  %-4zu %-7zu %-11s %+7.1f dB %5.1f ║\n",
                    m.regionIndex, m.lengthSamples,
                    severityToString(m.severity),
                    m.estimatedOvershootDb,
                    m.reconstructionSnrDb);
        }
    }

    fprintf(stderr, "╚══════════════════════════════════════════════╝\n\n");
}

}  // namespace faurge
