#include "faurge/saturator.hpp"
#include "faurge/sat_metrics.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void printUsage(const char* progName) {
    fprintf(stderr,
        "Faurge DSP \xe2\x80\x94 Harmonic Saturator v1.0\n"
        "Waveshaping saturator with 4 transfer functions for Ursula's RL training.\n\n"
        "Usage:\n"
        "  %s <input.wav> <output.wav> [options]\n\n"
        "Options:\n"
        "  --drive <db>        Drive 0..36                        (default: 0.0)\n"
        "  --type <str>        Sat type tube|tape|diode|asymmetric (default: tape)\n"
        "  --hpf <hz>          HPF cutoff 20..1000                 (default: 20.0)\n"
        "  --lpf <hz>          LPF cutoff 500..20000               (default: 20000.0)\n"
        "  --mix <0..1>        Wet/dry mix                         (default: 1.0)\n"
        "  --oversampling <n>  1|2                                  (default: 1)\n"
        "  --trim <db>         Output trim -12..+12                (default: 0.0)\n\n"
        "Global options:\n"
        "  --json              Output metrics as JSON to stdout\n"
        "  --verbose           Detailed processing log to stderr\n"
        "  --help              Print this message\n\n"
        "Examples:\n"
        "  %s input.wav output.wav --drive 6 --type tube --mix 0.8 --oversampling 2\n"
        "  %s input.wav output.wav --drive 12 --type asymmetric --hpf 200 --lpf 8000 --json\n\n",
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

    faurge::SatConfig config;

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

            if (arg == "--drive") {
                config.drive_db = parseFloat(0.0f, 36.0f);
            } else if (arg == "--hpf") {
                config.hpf_hz = parseFloat(20.0f, 1000.0f);
            } else if (arg == "--lpf") {
                config.lpf_hz = parseFloat(500.0f, 20000.0f);
            } else if (arg == "--mix") {
                config.mix = parseFloat(0.0f, 1.0f);
            } else if (arg == "--trim") {
                config.output_trim_db = parseFloat(-12.0f, 12.0f);
            } else if (arg == "--oversampling") {
                if (i + 1 >= argc) {
                    fprintf(stderr, "Missing value for: %s\n", argv[i]);
                    return 1;
                }
                int val = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
                if (val < 1) val = 1;
                if (val > 2) val = 2;
                config.oversampling = val;
            } else if (arg == "--type") {
                if (i + 1 >= argc) {
                    fprintf(stderr, "Missing value for: %s\n", argv[i]);
                    return 1;
                }
                std::string typeStr = argv[++i];
                if (typeStr == "tube") {
                    config.sat_type = static_cast<int>(faurge::SatType::tube);
                } else if (typeStr == "tape") {
                    config.sat_type = static_cast<int>(faurge::SatType::tape);
                } else if (typeStr == "diode") {
                    config.sat_type = static_cast<int>(faurge::SatType::diode);
                } else if (typeStr == "asymmetric") {
                    config.sat_type = static_cast<int>(faurge::SatType::asymmetric);
                } else {
                    fprintf(stderr, "Unknown sat type: %s\n", typeStr.c_str());
                    fprintf(stderr, "Valid types: tube, tape, diode, asymmetric\n");
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

    faurge::Saturator saturator(config);
    auto result = saturator.processFile(inputPath, outputPath);

    if (config.jsonOutput) {
        printf("%s", faurge::SatMetrics::toJson(result).c_str());
    } else {
        faurge::SatMetrics::printSummary(result);
    }

    return result.success ? 0 : 1;
}
