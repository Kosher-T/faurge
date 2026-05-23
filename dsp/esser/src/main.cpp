#include "faurge/esser.hpp"
#include "faurge/esser_metrics.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

static void printUsage(const char* prog) {
    fprintf(stderr,
        "Usage: %s input.wav output.wav [options]\n\n"
        "Options:\n"
        "  --center F      Center frequency in Hz (4000-10000, default 6000)\n"
        "  --threshold F   Threshold in dB (-60-0, default -30)\n"
        "  --ratio F       Ratio (0.25-20, default 5)\n"
        "  --bandwidth F   Bandwidth in Hz (500-4000, default 1500)\n"
        "  --attack F      Attack time in ms (0.1-50, default 2)\n"
        "  --release F     Release time in ms (10-500, default 100)\n"
        "  --json          Output JSON report\n"
        "  --verbose, -v   Verbose output\n"
        "  --help, -h      Show this help\n",
        prog);
}

static float clamp(float val, float lo, float hi) {
    return std::max(lo, std::min(hi, val));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    if (!std::strcmp(argv[1], "--help") || !std::strcmp(argv[1], "-h")) {
        printUsage(argv[0]);
        return 0;
    }

    const char* inputPath  = argv[1];
    const char* outputPath = argv[2];

    faurge::EsserConfig config;

    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--center") && i + 1 < argc) {
            config.center_freq_hz = clamp(std::atof(argv[++i]), 4000.0f, 10000.0f);
        } else if (!std::strcmp(argv[i], "--threshold") && i + 1 < argc) {
            config.threshold_db = clamp(std::atof(argv[++i]), -60.0f, 0.0f);
        } else if (!std::strcmp(argv[i], "--ratio") && i + 1 < argc) {
            config.ratio = clamp(std::atof(argv[++i]), 0.25f, 20.0f);
        } else if (!std::strcmp(argv[i], "--bandwidth") && i + 1 < argc) {
            config.bandwidth_hz = clamp(std::atof(argv[++i]), 500.0f, 4000.0f);
        } else if (!std::strcmp(argv[i], "--attack") && i + 1 < argc) {
            config.attack_ms = clamp(std::atof(argv[++i]), 0.1f, 50.0f);
        } else if (!std::strcmp(argv[i], "--release") && i + 1 < argc) {
            config.release_ms = clamp(std::atof(argv[++i]), 10.0f, 500.0f);
        } else if (!std::strcmp(argv[i], "--json")) {
            config.jsonOutput = true;
        } else if (!std::strcmp(argv[i], "--verbose") ||
                   !std::strcmp(argv[i], "-v")) {
            config.verbose = true;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    faurge::Esser esser(config);
    auto result = esser.processFile(inputPath, outputPath);

    if (config.jsonOutput) {
        fprintf(stdout, "%s", faurge::EsserMetrics::toJson(result).c_str());
    } else {
        faurge::EsserMetrics::printSummary(result);
        if (config.verbose) {
            fprintf(stderr, "[esser] Config: center=%g Hz, threshold=%g dB, "
                           "ratio=%g, bandwidth=%g Hz, attack=%g ms, "
                           "release=%g ms\n",
                    config.center_freq_hz, config.threshold_db, config.ratio,
                    config.bandwidth_hz, config.attack_ms, config.release_ms);
        }
    }

    return result.success ? 0 : 1;
}
