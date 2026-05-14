#include "faurge/denoiser.hpp"
#include "faurge/denoise_metrics.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void printUsage(const char* progName) {
    fprintf(stderr,
        "Faurge Specialist — Audio Denoiser v1.0\n"
        "Real-time noise and reverb suppression via DeepFilterNet 3.\n\n"
        "Usage:\n"
        "  %s <input.wav> <output.wav> [options]\n\n"
        "Options:\n"
        "  --atten-limit <float>   Suppression strength 0.0–1.0  (default: 0.78)\n"
        "  --model-path <path>     Path to DeepFilterNet model   (default: bundled)\n"
        "  --json                  Output metrics as JSON to stdout\n"
        "  --verbose               Detailed processing log to stderr\n"
        "  --help                  Print this message\n\n"
        "Examples:\n"
        "  %s noisy_speech.wav cleaned.wav --verbose\n"
        "  %s input.wav output.wav --atten-limit 0.9 --json\n\n",
        progName, progName, progName);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 ||
            std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        }
    }

    std::string inputPath  = argv[1];
    std::string outputPath = argv[2];

    faurge::DenoiseConfig config;

    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--atten-limit") == 0 && i + 1 < argc) {
            config.attenLimit = std::strtof(argv[++i], nullptr);
            if (config.attenLimit < 0.0f) config.attenLimit = 0.0f;
            if (config.attenLimit > 1.0f) config.attenLimit = 1.0f;
        } else if (std::strcmp(argv[i], "--model-path") == 0 && i + 1 < argc) {
            config.modelPath = argv[++i];
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

    faurge::Denoiser denoiser(config);
    auto result = denoiser.processFile(inputPath, outputPath);

    if (config.jsonOutput) {
        printf("%s", faurge::DenoiseMetrics::toJson(result).c_str());
    } else {
        faurge::DenoiseMetrics::printSummary(result);
    }

    return result.success ? 0 : 1;
}
