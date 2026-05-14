#include "faurge/denoise_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace faurge {

std::string DenoiseMetrics::toJson(const DenoiseResult& result) {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"success\": " << (result.success ? "true" : "false") << ",\n";
    ss << "  \"processing_time_ms\": " << result.processingTimeMs << ",\n";
    ss << "  \"input_snr_est_db\": " << result.inputSnrEstDb << ",\n";
    ss << "  \"output_snr_est_db\": " << result.outputSnrEstDb << ",\n";
    ss << "  \"noise_floor_dbfs\": " << result.noiseFloorDbfs << ",\n";
    ss << "  \"frames_processed\": " << result.framesProcessed << ",\n";
    ss << "  \"noise_reduction_ratio_db\": "
       << computeNoiseReductionRatioDb(result) << "\n";
    ss << "}\n";
    return ss.str();
}

void DenoiseMetrics::printSummary(const DenoiseResult& result) {
    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  FAURGE DENOISER — PROCESSING REPORT        ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Status:           %-25s ║\n",
            result.success ? "SUCCESS" : "FAILED");
    fprintf(stderr, "║  Processing time:  %-21.2f ms ║\n",
            result.processingTimeMs);
    fprintf(stderr, "║  Input SNR est:    %-21.1f dB ║\n",
            result.inputSnrEstDb);
    fprintf(stderr, "║  Output SNR est:   %-21.1f dB ║\n",
            result.outputSnrEstDb);
    fprintf(stderr, "║  Noise floor:      %-21.1f dBFS ║\n",
            result.noiseFloorDbfs);
    fprintf(stderr, "║  Frames processed: %-25zu ║\n",
            result.framesProcessed);
    fprintf(stderr, "║  NRR:              %-21.1f dB ║\n",
            computeNoiseReductionRatioDb(result));
    if (!result.errorMessage.empty()) {
        fprintf(stderr, "║  Error:            %-25s ║\n",
                result.errorMessage.c_str());
    }
    fprintf(stderr, "╚══════════════════════════════════════════════╝\n\n");
}

float DenoiseMetrics::computeNoiseReductionRatioDb(const DenoiseResult& result) {
    if (!result.success) return 0.0f;
    float nrr = result.inputSnrEstDb - result.outputSnrEstDb;
    return nrr;
}

}  // namespace faurge
