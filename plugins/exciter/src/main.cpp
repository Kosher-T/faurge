#include "faurge/exciter.hpp"
#include "faurge/exciter_metrics.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void printUsage(const char* progName) {
    fprintf(stderr,
        "Faurge Specialist \xe2\x80\x94 Audio Exciter v1.0\n"
        "Dual-band harmonic synthesis for bandwidth extension.\n\n"
        "Usage:\n"
        "  %s <input.wav> <output.wav> [options]\n\n"
        "Options:\n"
        "  --high-drive <float>  High-band pre-saturation gain dB  (default: 3.0)\n"
        "  --high-mix <float>    High-band wet/dry mix 0-1         (default: 0.50)\n"
        "  --high-cross <float>  High-band crossover frequency Hz  (default: 2000)\n"
        "  --low-drive <float>   Low-band pre-rectification gain dB (default: 0.0)\n"
        "  --low-mix <float>     Low-band wet/dry mix 0-1          (default: 0.35)\n"
        "  --low-cross <float>   Low-band crossover frequency Hz   (default: 200)\n"
        "  --low-sub <float>     Sub-octave injection level        (default: 0.50)\n"
        "  --no-high             Disable high-band processing\n"
        "  --no-low              Disable low-band processing\n"
        "  --volume <float>      Master volume                     (default: 1.0)\n"
        "  --json                Output metrics as JSON to stdout\n"
        "  --verbose             Detailed processing log to stderr\n"
        "  --help                Print this message\n\n"
        "Examples:\n"
        "  %s drum_loop.wav excited_loop.wav --high-drive 6 --low-mix 0.5\n"
        "  %s vocal.wav bright_vocal.wav --high-cross 3000 --json\n\n",
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

    faurge::ExciterConfig config;

    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--high-drive") == 0 && i + 1 < argc) {
            config.highDriveDb = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--high-mix") == 0 && i + 1 < argc) {
            config.highMix = std::strtof(argv[++i], nullptr);
            if (config.highMix < 0.0f) config.highMix = 0.0f;
            if (config.highMix > 1.0f) config.highMix = 1.0f;
        } else if (std::strcmp(argv[i], "--high-cross") == 0 && i + 1 < argc) {
            config.highCrossoverHz = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--low-drive") == 0 && i + 1 < argc) {
            config.lowDriveDb = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--low-mix") == 0 && i + 1 < argc) {
            config.lowMix = std::strtof(argv[++i], nullptr);
            if (config.lowMix < 0.0f) config.lowMix = 0.0f;
            if (config.lowMix > 1.0f) config.lowMix = 1.0f;
        } else if (std::strcmp(argv[i], "--low-cross") == 0 && i + 1 < argc) {
            config.lowCrossoverHz = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--low-sub") == 0 && i + 1 < argc) {
            config.lowSubLevel = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--no-high") == 0) {
            config.highEnable = false;
        } else if (std::strcmp(argv[i], "--no-low") == 0) {
            config.lowEnable = false;
        } else if (std::strcmp(argv[i], "--volume") == 0 && i + 1 < argc) {
            config.masterVolume = std::strtof(argv[++i], nullptr);
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

    faurge::Exciter exciter(config);
    auto result = exciter.processFile(inputPath, outputPath);

    if (config.jsonOutput) {
        printf("%s", faurge::ExciterMetrics::toJson(result).c_str());
    } else {
        faurge::ExciterMetrics::printSummary(result);
    }

    return result.success ? 0 : 1;
}
