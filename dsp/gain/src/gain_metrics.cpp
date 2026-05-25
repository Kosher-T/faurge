#include "faurge/gain_metrics.hpp"

#include <cmath>
#include <cstdio>
#include <sstream>

namespace faurge {

std::string GainMetrics::toJson(const GainResult& result) {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"success\": " << (result.success ? "true" : "false") << ",\n";
    ss << "  \"processing_time_ms\": " << result.processingTimeMs << ",\n";
    ss << "  \"input_peak_db\": " << result.inputPeakDb << ",\n";
    ss << "  \"output_peak_db\": " << result.outputPeakDb << ",\n";
    ss << "  \"input_rms_db\": " << result.inputRmsDb << ",\n";
    ss << "  \"output_rms_db\": " << result.outputRmsDb << ",\n";
    ss << "  \"input_lufs\": " << result.inputLufs << ",\n";
    ss << "  \"output_lufs\": " << result.outputLufs << ",\n";
    ss << "  \"peak_change_db\": " << result.peakChangeDb << ",\n";
    ss << "  \"rms_change_db\": " << result.rmsChangeDb << ",\n";
    ss << "  \"applied_balance\": " << result.appliedBalance << ",\n";
    ss << "  \"clipping\": " << (result.clipping ? "true" : "false") << ",\n";
    ss << "  \"frames_processed\": " << result.framesProcessed << "\n";
    ss << "}\n";
    return ss.str();
}

void GainMetrics::printSummary(const GainResult& result) {
    fprintf(stderr, "\n");
    fprintf(stderr,
        "\xe2\x95\x94\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
        "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
        "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
        "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
        "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
        "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
        "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x97\n");
    fprintf(stderr,
        "\xe2\x95\x91  FAURGE GAIN \xe2\x80\x94 LEVEL & BALANCE REPORT "
        "\xe2\x95\x91\n");
    fprintf(stderr,
        "\xe2\x95\xa0\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
        "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
        "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
        "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
        "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
        "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
        "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\xa3\n");
    fprintf(stderr, "\xe2\x95\x91  Status:              %-21s \xe2\x95\x91\n",
            result.success ? "SUCCESS" : "FAILED");
    fprintf(stderr, "\xe2\x95\x91  Processing time:     %-17.2f ms \xe2\x95\x91\n",
            result.processingTimeMs);
    fprintf(stderr, "\xe2\x95\x91  Gain:                %-17.1f dB \xe2\x95\x91\n",
            result.peakChangeDb);
    fprintf(stderr, "\xe2\x95\x91  Balance:             %-17.1f   \xe2\x95\x91\n",
            result.appliedBalance);
    fprintf(stderr, "\xe2\x95\x91  Input peak:          %-17.1f dB \xe2\x95\x91\n",
            result.inputPeakDb);
    fprintf(stderr, "\xe2\x95\x91  Output peak:         %-17.1f dB \xe2\x95\x91\n",
            result.outputPeakDb);
    fprintf(stderr, "\xe2\x95\x91  Input RMS:           %-17.1f dB \xe2\x95\x91\n",
            result.inputRmsDb);
    fprintf(stderr, "\xe2\x95\x91  Output RMS:          %-17.1f dB \xe2\x95\x91\n",
            result.outputRmsDb);
    fprintf(stderr, "\xe2\x95\x91  Input LUFS:          %-17.1f    \xe2\x95\x91\n",
            result.inputLufs);
    fprintf(stderr, "\xe2\x95\x91  Output LUFS:         %-17.1f    \xe2\x95\x91\n",
            result.outputLufs);
    fprintf(stderr, "\xe2\x95\x91  Peak change:         %-17.1f dB \xe2\x95\x91\n",
            result.peakChangeDb);
    fprintf(stderr, "\xe2\x95\x91  RMS change:          %-17.1f dB \xe2\x95\x91\n",
            result.rmsChangeDb);
    fprintf(stderr, "\xe2\x95\x91  Clipping:            %-21s \xe2\x95\x91\n",
            result.clipping ? "YES" : "no");
    fprintf(stderr, "\xe2\x95\x91  Frames processed:    %-21zu \xe2\x95\x91\n",
            result.framesProcessed);
    fprintf(stderr,
        "\xe2\x95\x9a\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
        "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
        "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
        "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
        "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
        "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
        "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x9d\n\n");
}

} // namespace faurge
