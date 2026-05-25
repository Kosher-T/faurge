#include "faurge/transient.hpp"
#include "faurge/transient_metrics.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void printUsage(const char* progName) {
    fprintf(stderr,
        "Faurge DSP \xe2\x80\x94 Transient Shaper v1.0\n"
        "Attack/sustain envelope separation for Ursula's RL training and inference.\n\n"
        "Usage:\n"
        "  %s <input.wav> <output.wav> [options]\n\n"
        "Options:\n"
        "  --attack-gain <db>  Attack gain -24..+24          (default: 6.0)\n"
        "  --sustain-gain <db> Sustain gain -24..+24         (default: 0.0)\n"
        "  --attack-time <ms>  Attack time 0.1..50           (default: 5.0)\n"
        "  --release-time <ms> Release time 10..500          (default: 100.0)\n"
        "  --sensitivity <db>  Sensitivity threshold -30..0  (default: -30.0)\n"
        "  --mix <0..1>        Wet/dry mix                   (default: 1.0)\n\n"
        "Global options:\n"
        "  --json              Output metrics as JSON to stdout\n"
        "  --verbose           Detailed processing log to stderr\n"
        "  --help              Print this message\n\n"
        "Examples:\n"
        "  %s drums.wav out.wav --attack-gain 12 --sustain-gain -6 --attack-time 3 --release-time 200\n"
        "  %s vocal.wav out.wav --attack-gain -6 --sustain-gain 3 --mix 0.7 --json\n\n",
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

    faurge::TransientConfig config;

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

            if (arg == "--attack-gain") {
                config.attack_gain_db = parseFloat(-24.0f, 24.0f);
            } else if (arg == "--sustain-gain") {
                config.sustain_gain_db = parseFloat(-24.0f, 24.0f);
            } else if (arg == "--attack-time") {
                config.attack_time_ms = parseFloat(0.1f, 50.0f);
            } else if (arg == "--release-time") {
                config.release_time_ms = parseFloat(10.0f, 500.0f);
            } else if (arg == "--sensitivity") {
                config.sensitivity_db = parseFloat(-30.0f, 0.0f);
            } else if (arg == "--mix") {
                config.mix = parseFloat(0.0f, 1.0f);
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

    faurge::TransientShaper shaper(config);
    auto result = shaper.processFile(inputPath, outputPath);

    if (config.jsonOutput) {
        printf("%s", faurge::TransientMetrics::toJson(result).c_str());
    } else {
        faurge::TransientMetrics::printSummary(result);
    }

    return result.success ? 0 : 1;
}
