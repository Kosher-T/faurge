#include "faurge/gain.hpp"
#include "faurge/gain_metrics.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void printUsage(const char* progName) {
    fprintf(stderr,
        "Faurge DSP \xe2\x80\x94 Gain v1.0\n"
        "Level trim + stereo balance for Ursula's RL training and inference.\n\n"
        "Usage:\n"
        "  %s <input.wav> <output.wav> [options]\n\n"
        "Options:\n"
        "  --gain-db <db>   Gain -12..+12 dB            (default: 0.0)\n"
        "  --balance <val>  Stereo balance -1..1         (default: 0.0)\n\n"
        "Global options:\n"
        "  --json              Output metrics as JSON to stdout\n"
        "  --verbose           Detailed processing log to stderr\n"
        "  --help              Print this message\n\n"
        "Examples:\n"
        "  %s speech.wav out.wav --gain-db -6\n"
        "  %s mix.wav out.wav --gain-db 3 --balance -0.5 --json\n\n",
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

    faurge::GainConfig config;

    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--json") == 0) {
            config.jsonOutput = true;
        } else if (std::strcmp(argv[i], "--verbose") == 0) {
            config.verbose = true;
        } else if (argv[i][0] == '-' && argv[i][1] == '-') {
            std::string arg = argv[i];

            auto parseFloat = [&](float lo, float hi) -> float {
                if (i + 1 >= argc) {
                    fprintf(stderr, "Missing value for: %s\n", argv[i]);
                    std::exit(1);
                }
                float val = std::strtof(argv[++i], nullptr);
                if (val < lo) val = lo;
                if (val > hi) val = hi;
                return val;
            };

            if (arg == "--gain-db") {
                config.gain_db = parseFloat(-12.0f, 12.0f);
            } else if (arg == "--balance") {
                config.stereo_balance = parseFloat(-1.0f, 1.0f);
            } else {
                fprintf(stderr, "Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                return 1;
            }
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    faurge::Gain gainer(config);
    auto result = gainer.processFile(inputPath, outputPath);

    if (config.jsonOutput) {
        printf("%s", faurge::GainMetrics::toJson(result).c_str());
    } else {
        faurge::GainMetrics::printSummary(result);
    }

    return result.success ? 0 : 1;
}
