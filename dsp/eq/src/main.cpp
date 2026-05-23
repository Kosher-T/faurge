#include "faurge/eq.hpp"
#include "faurge/eq_metrics.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void printUsage(const char* progName) {
    fprintf(stderr,
        "Faurge DSP \xe2\x80\x94 31-Band Parametric EQ v1.0\n"
        "Spectral shaping tool for Ursula's RL training and inference.\n\n"
        "Usage:\n"
        "  %s <input.wav> <output.wav> [options]\n\n"
        "Options (per band, 1..31):\n"
        "  --band<N>-freq <hz>     Center frequency 20-20000      (default: 1000)\n"
        "  --band<N>-gain <db>     Gain -24..+24                  (default: 0.0)\n"
        "  --band<N>-q <float>     Q factor 0.1-10               (default: 1.0)\n"
        "  --band<N>-type <str>    Filter type: peak|low_shelf|high_shelf|\n"
        "                          highpass|lowpass|bandpass|notch (default: peak)\n"
        "  --band<N>-skew <db>     L/R gain difference -6..+6    (default: 0.0)\n"
        "  --band<N>-dynamic <0|1> Dynamic depth 0-1             (default: 0.0)\n\n"
        "Global options:\n"
        "  --json                  Output metrics as JSON to stdout\n"
        "  --verbose               Detailed processing log to stderr\n"
        "  --help                  Print this message\n\n"
        "Examples:\n"
        "  %s input.wav output.wav --band1-freq 1000 --band1-gain -3 --band1-q 1.4 --band1-type peak\n"
        "  %s input.wav output.wav --band1-freq 200 --band1-gain +2.5 --band1-q 0.7 --band1-type low_shelf --json\n\n",
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

    faurge::EqConfig config;

    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--json") == 0) {
            config.jsonOutput = true;
        } else if (std::strcmp(argv[i], "--verbose") == 0) {
            config.verbose = true;
        } else if (argv[i][0] == '-' && argv[i][1] == '-') {
            std::string arg = argv[i];
            int bandIdx = -1;

            if (arg.find("--band") != 0) {
                fprintf(stderr, "Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                return 1;
            }

            size_t prefixLen = 6;
            size_t digitEnd = prefixLen;
            while (digitEnd < arg.size() && std::isdigit(arg[digitEnd])) {
                ++digitEnd;
            }
            if (digitEnd == prefixLen) {
                fprintf(stderr, "Invalid band index in: %s\n", argv[i]);
                return 1;
            }

            bandIdx = std::stoi(arg.substr(prefixLen, digitEnd - prefixLen));
            if (bandIdx < 1 || bandIdx > faurge::NUM_EQ_BANDS) {
                fprintf(stderr, "Band index out of range (1-%d): %s\n",
                        faurge::NUM_EQ_BANDS, argv[i]);
                return 1;
            }

            std::string param = arg.substr(digitEnd);

            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for: %s\n", argv[i]);
                return 1;
            }

            faurge::FilterBand& band = config.bands[bandIdx - 1];

            if (param == "-freq") {
                band.freq_hz = std::strtof(argv[++i], nullptr);
                if (band.freq_hz < 20.0f) band.freq_hz = 20.0f;
                if (band.freq_hz > 20000.0f) band.freq_hz = 20000.0f;
            } else if (param == "-gain") {
                band.gain_db = std::strtof(argv[++i], nullptr);
                if (band.gain_db < -24.0f) band.gain_db = -24.0f;
                if (band.gain_db > 24.0f) band.gain_db = 24.0f;
            } else if (param == "-q") {
                band.q = std::strtof(argv[++i], nullptr);
                if (band.q < 0.1f) band.q = 0.1f;
                if (band.q > 10.0f) band.q = 10.0f;
            } else if (param == "-type") {
                std::string typeStr = argv[++i];
                if (typeStr == "peak")          band.filter_type = faurge::FilterType::peak;
                else if (typeStr == "low_shelf")   band.filter_type = faurge::FilterType::low_shelf;
                else if (typeStr == "high_shelf")  band.filter_type = faurge::FilterType::high_shelf;
                else if (typeStr == "highpass")    band.filter_type = faurge::FilterType::highpass;
                else if (typeStr == "lowpass")     band.filter_type = faurge::FilterType::lowpass;
                else if (typeStr == "bandpass")    band.filter_type = faurge::FilterType::bandpass;
                else if (typeStr == "notch")       band.filter_type = faurge::FilterType::notch;
                else {
                    fprintf(stderr, "Unknown filter type: %s\n", typeStr.c_str());
                    return 1;
                }
            } else if (param == "-skew") {
                band.stereo_skew_db = std::strtof(argv[++i], nullptr);
                if (band.stereo_skew_db < -6.0f) band.stereo_skew_db = -6.0f;
                if (band.stereo_skew_db > 6.0f) band.stereo_skew_db = 6.0f;
            } else if (param == "-dynamic") {
                band.dynamic_depth = std::strtof(argv[++i], nullptr);
                if (band.dynamic_depth < 0.0f) band.dynamic_depth = 0.0f;
                if (band.dynamic_depth > 1.0f) band.dynamic_depth = 1.0f;
            } else {
                fprintf(stderr, "Unknown parameter: %s\n", argv[i]);
                return 1;
            }
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    faurge::Equalizer eq(config);
    auto result = eq.processFile(inputPath, outputPath);

    if (config.jsonOutput) {
        printf("%s", faurge::EqMetrics::toJson(result).c_str());
    } else {
        faurge::EqMetrics::printSummary(result);
    }

    return result.success ? 0 : 1;
}
