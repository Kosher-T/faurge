#include "faurge/limiter.hpp"
#include "faurge/limiter_metrics.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void printUsage(const char* progName) {
    fprintf(stderr,
        "Faurge DSP \xe2\x80\x94 Peak Limiter v1.0\n"
        "Brickwall peak limiter with lookahead for Ursula's RL training and inference.\n\n"
        "Usage:\n"
        "  %s <input.wav> <output.wav> [options]\n\n"
        "Options:\n"
        "  --ceiling <db>      Output ceiling -12..0            (default: -1.0)\n"
        "  --release <ms>      Release time 1..500              (default: 100.0)\n"
        "  --lookahead <ms>    Lookahead 0..10                  (default: 5.0)\n"
        "  --clip-mode <str>   hard|soft                        (default: soft)\n"
        "  --link <0..1>       Stereo link                       (default: 1.0)\n"
        "  --oversampling <n>  1|2|4|8                           (default: 1)\n\n"
        "Global options:\n"
        "  --json              Output metrics as JSON to stdout\n"
        "  --verbose           Detailed processing log to stderr\n"
        "  --help              Print this message\n\n"
        "Examples:\n"
        "  %s input.wav output.wav --ceiling -3 --release 50 --lookahead 3 --clip-mode soft\n"
        "  %s input.wav output.wav --ceiling -6 --lookahead 5 --clip-mode hard --oversampling 4 --json\n\n",
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

    faurge::LimiterConfig config;

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

            auto parseInt = [&](int lo, int hi) -> int {
                if (i + 1 >= argc) {
                    fprintf(stderr, "Missing value for: %s\n", argv[i]);
                    std::exit(1);
                }
                int val = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
                if (val < lo) val = lo;
                if (val > hi) val = hi;
                return val;
            };

            if (arg == "--ceiling") {
                config.ceiling_db = parseFloat(-12.0f, 0.0f);
            } else if (arg == "--release") {
                config.release_ms = parseFloat(1.0f, 500.0f);
            } else if (arg == "--lookahead") {
                config.lookahead_ms = parseFloat(0.0f, 10.0f);
            } else if (arg == "--link") {
                config.stereo_link = parseFloat(0.0f, 1.0f);
            } else if (arg == "--oversampling") {
                config.oversampling = parseInt(1, 8);
            } else if (arg == "--clip-mode") {
                if (i + 1 >= argc) {
                    fprintf(stderr, "Missing value for: %s\n", argv[i]);
                    return 1;
                }
                std::string modeStr = argv[++i];
                if (modeStr == "hard") {
                    config.clip_mode = faurge::ClipMode::hard;
                } else if (modeStr == "soft") {
                    config.clip_mode = faurge::ClipMode::soft;
                } else {
                    fprintf(stderr, "Unknown clip mode: %s\n", modeStr.c_str());
                    fprintf(stderr, "Valid modes: hard, soft\n");
                    return 1;
                }
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

    faurge::Limiter limiter(config);
    auto result = limiter.processFile(inputPath, outputPath);

    if (config.jsonOutput) {
        printf("%s", faurge::LimiterMetrics::toJson(result).c_str());
    } else {
        faurge::LimiterMetrics::printSummary(result);
    }

    return result.success ? 0 : 1;
}
