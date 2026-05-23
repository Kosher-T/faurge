#include "faurge/compressor.hpp"
#include "faurge/comp_metrics.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void printUsage(const char* progName) {
    fprintf(stderr,
        "Faurge DSP \xe2\x80\x94 Dynamic Range Compressor v1.0\n"
        "Dynamic range control tool for Ursula's RL training and inference.\n\n"
        "Usage:\n"
        "  %s <input.wav> <output.wav> [options]\n\n"
        "Options:\n"
        "  --threshold <db>    Threshold level -60..0            (default: -24.0)\n"
        "  --ratio <float>     Compression ratio 1..20           (default: 4.0)\n"
        "  --attack <ms>       Attack time 0.1..100              (default: 5.0)\n"
        "  --release <ms>      Release time 10..1000             (default: 150.0)\n"
        "  --knee <db>         Knee width 0..12                  (default: 6.0)\n"
        "  --lookahead <ms>    Lookahead 0..10                   (default: 0.0)\n"
        "  --hold <ms>         Hold time 0..200                  (default: 0.0)\n"
        "  --mix <0..1>        Wet/dry mix                       (default: 1.0)\n"
        "  --link <0..1>       Stereo link                       (default: 1.0)\n"
        "  --sidechain-hp <hz> Sidechain HP filter 20..500       (default: 20.0)\n"
        "  --sidechain-lp <hz> Sidechain LP filter 500..20000    (default: 20000.0)\n"
        "  --saturate <db>     Saturation drive 0..12            (default: 0.0)\n"
        "  --trim <db>         Output trim -12..+12              (default: 0.0)\n"
        "  --detector <str>    RMS|peak|feed_forward|feed_back   (default: RMS)\n\n"
        "Global options:\n"
        "  --json              Output metrics as JSON to stdout\n"
        "  --verbose           Detailed processing log to stderr\n"
        "  --help              Print this message\n\n"
        "Examples:\n"
        "  %s input.wav output.wav --threshold -24 --ratio 4 --attack 5 --release 150 --knee 6\n"
        "  %s input.wav output.wav --threshold -30 --ratio 8 --attack 2 --release 200 --json\n\n",
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

    faurge::CompConfig config;

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

            if (arg == "--threshold") {
                config.threshold_db = parseFloat(-60.0f, 0.0f);
            } else if (arg == "--ratio") {
                config.ratio = parseFloat(1.0f, 20.0f);
            } else if (arg == "--attack") {
                config.attack_ms = parseFloat(0.1f, 100.0f);
            } else if (arg == "--release") {
                config.release_ms = parseFloat(10.0f, 1000.0f);
            } else if (arg == "--knee") {
                config.knee_db = parseFloat(0.0f, 12.0f);
            } else if (arg == "--lookahead") {
                config.lookahead_ms = parseFloat(0.0f, 10.0f);
            } else if (arg == "--hold") {
                config.hold_ms = parseFloat(0.0f, 200.0f);
            } else if (arg == "--mix") {
                config.wet_dry_mix = parseFloat(0.0f, 1.0f);
            } else if (arg == "--link") {
                config.stereo_link = parseFloat(0.0f, 1.0f);
            } else if (arg == "--sidechain-hp") {
                config.sidechain_hp_hz = parseFloat(20.0f, 500.0f);
            } else if (arg == "--sidechain-lp") {
                config.sidechain_lp_hz = parseFloat(500.0f, 20000.0f);
            } else if (arg == "--saturate") {
                config.saturate_drive_db = parseFloat(0.0f, 12.0f);
            } else if (arg == "--trim") {
                config.output_trim_db = parseFloat(-12.0f, 12.0f);
            } else if (arg == "--detector") {
                if (i + 1 >= argc) {
                    fprintf(stderr, "Missing value for: %s\n", argv[i]);
                    return 1;
                }
                std::string typeStr = argv[++i];
                if (typeStr == "RMS" || typeStr == "rms") {
                    config.detector_type = faurge::DetectorType::RMS;
                } else if (typeStr == "peak") {
                    config.detector_type = faurge::DetectorType::peak;
                } else if (typeStr == "feed_forward") {
                    config.detector_type = faurge::DetectorType::feed_forward;
                } else if (typeStr == "feed_back") {
                    config.detector_type = faurge::DetectorType::feed_back;
                } else {
                    fprintf(stderr, "Unknown detector type: %s\n", typeStr.c_str());
                    fprintf(stderr, "Valid types: RMS, peak, feed_forward, feed_back\n");
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

    faurge::Compressor compressor(config);
    auto result = compressor.processFile(inputPath, outputPath);

    if (config.jsonOutput) {
        printf("%s", faurge::CompMetrics::toJson(result).c_str());
    } else {
        faurge::CompMetrics::printSummary(result);
    }

    return result.success ? 0 : 1;
}
