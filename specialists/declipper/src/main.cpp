// ═══════════════════════════════════════════════════════════════
// Faurge Specialist — De-Clipper: CLI Entry Point
// ═══════════════════════════════════════════════════════════════
// Usage:
//   faurge-declip input.wav output.wav [options]
//
// Options:
//   --threshold <float>   Clip detection threshold (default: 0.9999)
//   --merge-gap <int>     Max gap between clips to merge (default: 3)
//   --anchor-size <int>   Boundary anchor samples (default: 4)
//   --crossfade <int>     Crossfade width in samples (default: 8)
//   --no-soft-clip        Disable soft-clip detection
//   --json                Output metrics as JSON to stdout
//   --verbose             Detailed per-region logging to stderr
//   --help                Print this message
// ═══════════════════════════════════════════════════════════════
#include "faurge/declipper.hpp"
#include "faurge/metrics.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void printUsage(const char* progName) {
    fprintf(stderr,
        "Faurge Specialist — Audio De-Clipper v1.0\n"
        "Reconstructs clipped waveforms via polynomial interpolation.\n\n"
        "Usage:\n"
        "  %s <input.wav> <output.wav> [options]\n\n"
        "Options:\n"
        "  --threshold <float>   Clip detection threshold    (default: 0.9999)\n"
        "  --merge-gap <int>     Max merge gap in samples    (default: 3)\n"
        "  --anchor-size <int>   Boundary context samples    (default: 4)\n"
        "  --crossfade <int>     Crossfade width in samples  (default: 8)\n"
        "  --overshoot <float>   Peak overshoot multiplier   (default: 1.15)\n"
        "  --ar-order <int>      AR model order (long clips) (default: 14)\n"
        "  --no-soft-clip        Disable soft-clip detection\n"
        "  --no-anti-alias       Disable anti-alias filter\n"
        "  --json                Output metrics as JSON to stdout\n"
        "  --verbose             Detailed per-region logging\n"
        "  --help                Print this message\n\n"
        "Examples:\n"
        "  %s clipped_sermon.wav clean_sermon.wav --verbose\n"
        "  %s noisy.wav fixed.wav --threshold 0.999 --json\n\n",
        progName, progName, progName);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    // Check for --help first
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 ||
            std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        }
    }

    std::string inputPath  = argv[1];
    std::string outputPath = argv[2];

    faurge::DeclipConfig config;

    // Parse optional arguments
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            config.clipThreshold = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--merge-gap") == 0 && i + 1 < argc) {
            config.mergeGap = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--anchor-size") == 0 && i + 1 < argc) {
            config.anchorSize = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--crossfade") == 0 && i + 1 < argc) {
            config.crossfadeWidth = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--overshoot") == 0 && i + 1 < argc) {
            config.peakOvershoot = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--ar-order") == 0 && i + 1 < argc) {
            config.arModelOrder = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--no-soft-clip") == 0) {
            config.detectSoftClip = false;
        } else if (std::strcmp(argv[i], "--no-anti-alias") == 0) {
            config.enableAntiAlias = false;
        } else if (std::strcmp(argv[i], "--json") == 0) {
            config.jsonOutput = true;
        } else if (std::strcmp(argv[i], "--verbose") == 0) {
            config.verbose = true;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    // Run the de-clipper
    faurge::Declipper declipper(config);
    auto result = declipper.processFile(inputPath, outputPath);

    // Output results
    if (config.jsonOutput) {
        printf("%s", faurge::Metrics::toJson(result).c_str());
    } else {
        faurge::Metrics::printSummary(result);
    }

    return result.success ? 0 : 1;
}
