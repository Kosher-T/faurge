// ═══════════════════════════════════════════════════════════════
// Faurge — Exciter Training Pair Generator
// ═══════════════════════════════════════════════════════════════
// Usage:
//   faurge-generate-excite-pairs <clean.wav> <excited.wav> [options]
//
// Applies controlled harmonic distortion and sub-octave synthesis
// to create degraded/excited pairs for cloud training.
// ═══════════════════════════════════════════════════════════════

#include <sndfile.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static constexpr float PI = 3.14159265358979f;

struct Config {
    float highDriveDb     = 6.0f;
    float highMix         = 0.5f;
    float highCrossoverHz = 2000.0f;
    float lowDriveDb      = 0.0f;
    float lowMix          = 0.3f;
    float lowCrossoverHz  = 200.0f;
    float lowSubLevel     = 0.5f;
    bool  highEnable      = true;
    bool  lowEnable       = true;
    float masterVolume    = 1.0f;
    bool  verbose         = false;
};

static void printUsage(const char* progName) {
    fprintf(stderr,
        "Faurge — Exciter Training Pair Generator\n\n"
        "Usage:\n"
        "  %s <clean.wav> <excited.wav> [options]\n\n"
        "Options:\n"
        "  --high-drive <float>  High-band drive dB     (default: 6.0)\n"
        "  --high-mix <float>    High-band mix 0-1      (default: 0.5)\n"
        "  --high-cross <float>  High crossover Hz      (default: 2000)\n"
        "  --low-drive <float>   Low-band drive dB      (default: 0.0)\n"
        "  --low-mix <float>     Low-band mix 0-1       (default: 0.3)\n"
        "  --low-cross <float>   Low crossover Hz       (default: 200)\n"
        "  --low-sub <float>     Sub-octave level       (default: 0.5)\n"
        "  --no-high             Disable high band\n"
        "  --no-low              Disable low band\n"
        "  --volume <float>      Master volume          (default: 1.0)\n"
        "  --verbose             Verbose output\n"
        "  --help                Print this message\n",
        progName);
}

struct BiquadState { float x1 = 0.0f, x2 = 0.0f, y1 = 0.0f, y2 = 0.0f; };

static void applyLpfCascade(const float* in, float* out, size_t n,
                            int sampleRate, float freqHz,
                            BiquadState& s1, BiquadState& s2) {
    float w0 = 2.0f * PI * freqHz / static_cast<float>(sampleRate);
    float alpha = std::sin(w0) / 1.41421356f;
    float b0 = (1.0f - std::cos(w0)) / 2.0f;
    float b1 = 1.0f - std::cos(w0);
    float b2 = b0;
    float a0 = 1.0f + alpha;
    float a1 = -2.0f * std::cos(w0);
    float a2 = 1.0f - alpha;
    b0 /= a0; b1 /= a0; b2 /= a0; a1 /= a0; a2 /= a0;

    std::vector<float> tmp(n);
    for (size_t i = 0; i < n; ++i) {
        float y = b0 * in[i] + b1 * s1.x1 + b2 * s1.x2 - a1 * s1.y1 - a2 * s1.y2;
        s1.x2 = s1.x1; s1.x1 = in[i];
        s1.y2 = s1.y1; s1.y1 = y;
        tmp[i] = y;
    }
    for (size_t i = 0; i < n; ++i) {
        float y = b0 * tmp[i] + b1 * s2.x1 + b2 * s2.x2 - a1 * s2.y1 - a2 * s2.y2;
        s2.x2 = s2.x1; s2.x1 = tmp[i];
        s2.y2 = s2.y1; s2.y1 = y;
        out[i] = y;
    }
}

struct LR4State {
    BiquadState lp1, lp2, hp1, hp2;
};

static void lr4Split(const float* in, float* low, float* high, size_t n,
                     int sr, float freq, LR4State& s) {
    if (freq <= 20.0f) {
        std::memcpy(low, in, n * sizeof(float));
        std::memset(high, 0, n * sizeof(float));
        return;
    }
    float nyq = 0.5f * sr;
    if (freq >= nyq) {
        std::memset(low, 0, n * sizeof(float));
        std::memcpy(high, in, n * sizeof(float));
        return;
    }

    float w0 = 2.0f * PI * freq / sr;
    float alpha = std::sin(w0) / 1.41421356f;

    float bl0 = (1.0f - std::cos(w0)) / 2.0f;
    float bl1 = 1.0f - std::cos(w0);
    float bl2 = bl0;
    float al0 = 1.0f + alpha;
    float al1 = -2.0f * std::cos(w0);
    float al2 = 1.0f - alpha;
    bl0 /= al0; bl1 /= al0; bl2 /= al0; al1 /= al0; al2 /= al0;

    float bh0 = (1.0f + std::cos(w0)) / 2.0f;
    float bh1 = -(1.0f + std::cos(w0));
    float bh2 = bh0;
    float ah0 = 1.0f + alpha;
    float ah1 = -2.0f * std::cos(w0);
    float ah2 = 1.0f - alpha;
    bh0 /= ah0; bh1 /= ah0; bh2 /= ah0; ah1 /= ah0; ah2 /= ah0;

    std::vector<float> tmp(n);

    for (size_t i = 0; i < n; ++i) {
        float y = bl0 * in[i] + bl1 * s.lp1.x1 + bl2 * s.lp1.x2
                - al1 * s.lp1.y1 - al2 * s.lp1.y2;
        s.lp1.x2 = s.lp1.x1; s.lp1.x1 = in[i];
        s.lp1.y2 = s.lp1.y1; s.lp1.y1 = y;
        tmp[i] = y;
    }
    for (size_t i = 0; i < n; ++i) {
        float y = bl0 * tmp[i] + bl1 * s.lp2.x1 + bl2 * s.lp2.x2
                - al1 * s.lp2.y1 - al2 * s.lp2.y2;
        s.lp2.x2 = s.lp2.x1; s.lp2.x1 = tmp[i];
        s.lp2.y2 = s.lp2.y1; s.lp2.y1 = y;
        low[i] = y;
    }

    for (size_t i = 0; i < n; ++i) {
        float y = bh0 * in[i] + bh1 * s.hp1.x1 + bh2 * s.hp1.x2
                - ah1 * s.hp1.y1 - ah2 * s.hp1.y2;
        s.hp1.x2 = s.hp1.x1; s.hp1.x1 = in[i];
        s.hp1.y2 = s.hp1.y1; s.hp1.y1 = y;
        tmp[i] = y;
    }
    for (size_t i = 0; i < n; ++i) {
        float y = bh0 * tmp[i] + bh1 * s.hp2.x1 + bh2 * s.hp2.x2
                - ah1 * s.hp2.y1 - ah2 * s.hp2.y2;
        s.hp2.x2 = s.hp2.x1; s.hp2.x1 = tmp[i];
        s.hp2.y2 = s.hp2.y1; s.hp2.y1 = y;
        high[i] = y;
    }
}

static void processChannel(std::vector<float>& audio, int sampleRate,
                           const Config& cfg) {
    size_t n = audio.size();
    std::vector<float> highBuf(n), lowBuf(n);
    std::vector<float> highOut(n), lowOut(n);

    LR4State hcState{}, lcState{};
    lr4Split(audio.data(), nullptr, highBuf.data(), n, sampleRate,
             cfg.highCrossoverHz, hcState);
    lr4Split(audio.data(), lowBuf.data(), nullptr, n, sampleRate,
             cfg.lowCrossoverHz, lcState);

    if (cfg.highEnable) {
        float drive = std::pow(10.0f, cfg.highDriveDb / 20.0f);
        std::vector<float> osBuf(n * 2);
        for (size_t i = 0; i < n; ++i) {
            osBuf[i * 2] = highBuf[i];
            osBuf[i * 2 + 1] = highBuf[i];
        }
        for (size_t i = 0; i < n * 2; ++i) {
            osBuf[i] = std::tanh(osBuf[i] * drive);
        }
        for (size_t i = 0; i < n; ++i) {
            highOut[i] = osBuf[i * 2];
        }
    } else {
        std::memcpy(highOut.data(), highBuf.data(), n * sizeof(float));
    }

    if (cfg.lowEnable) {
        float drive = std::pow(10.0f, cfg.lowDriveDb / 20.0f);
        BiquadState lpf1{}, lpf2{};
        std::vector<float> rect(n);
        for (size_t i = 0; i < n; ++i) {
            rect[i] = std::fabs(lowBuf[i]) * drive;
        }
        applyLpfCascade(rect.data(), lowOut.data(), n, sampleRate,
                        120.0f, lpf1, lpf2);
        for (size_t i = 0; i < n; ++i) {
            lowOut[i] *= cfg.lowSubLevel;
        }
    } else {
        std::memcpy(lowOut.data(), lowBuf.data(), n * sizeof(float));
    }

    for (size_t i = 0; i < n; ++i) {
        float highDelta = highOut[i] - highBuf[i];
        float lowDelta = lowOut[i] - lowBuf[i];
        audio[i] = (audio[i] + cfg.highMix * highDelta
                           + cfg.lowMix * lowDelta) * cfg.masterVolume;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        }
    }

    std::string inputPath  = argv[1];
    std::string outputPath = argv[2];

    Config cfg;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--high-drive") == 0 && i + 1 < argc)
            cfg.highDriveDb = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--high-mix") == 0 && i + 1 < argc)
            cfg.highMix = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--high-cross") == 0 && i + 1 < argc)
            cfg.highCrossoverHz = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--low-drive") == 0 && i + 1 < argc)
            cfg.lowDriveDb = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--low-mix") == 0 && i + 1 < argc)
            cfg.lowMix = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--low-cross") == 0 && i + 1 < argc)
            cfg.lowCrossoverHz = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--low-sub") == 0 && i + 1 < argc)
            cfg.lowSubLevel = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--no-high") == 0)
            cfg.highEnable = false;
        else if (std::strcmp(argv[i], "--no-low") == 0)
            cfg.lowEnable = false;
        else if (std::strcmp(argv[i], "--volume") == 0 && i + 1 < argc)
            cfg.masterVolume = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--verbose") == 0)
            cfg.verbose = true;
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    SF_INFO sfInfoIn;
    std::memset(&sfInfoIn, 0, sizeof(sfInfoIn));
    SNDFILE* inFile = sf_open(inputPath.c_str(), SFM_READ, &sfInfoIn);
    if (!inFile) {
        fprintf(stderr, "Failed to open input: %s\n", inputPath.c_str());
        return 1;
    }

    int channels = sfInfoIn.channels;
    int sr = sfInfoIn.samplerate;
    sf_count_t frames = sfInfoIn.frames;
    size_t totalSamples = static_cast<size_t>(frames) * channels;
    std::vector<float> buf(totalSamples);

    sf_count_t read = sf_readf_float(inFile, buf.data(), frames);
    sf_close(inFile);
    if (read != frames) {
        fprintf(stderr, "Failed to read all frames\n");
        return 1;
    }

    if (cfg.verbose) {
        fprintf(stderr, "[generate] Input: %s (%d ch, %d Hz, %lld frames)\n",
                inputPath.c_str(), channels, sr,
                static_cast<long long>(frames));
    }

    std::vector<std::vector<float>> chBufs(channels);
    for (int c = 0; c < channels; ++c) {
        chBufs[c].resize(frames);
        for (sf_count_t f = 0; f < frames; ++f)
            chBufs[c][f] = buf[f * channels + c];
    }

    for (int c = 0; c < channels; ++c) {
        processChannel(chBufs[c], sr, cfg);
    }

    for (int c = 0; c < channels; ++c) {
        for (sf_count_t f = 0; f < frames; ++f)
            buf[f * channels + c] = chBufs[c][f];
    }

    SF_INFO sfInfoOut = sfInfoIn;
    sfInfoOut.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    SNDFILE* outFile = sf_open(outputPath.c_str(), SFM_WRITE, &sfInfoOut);
    if (!outFile) {
        fprintf(stderr, "Failed to open output: %s\n", outputPath.c_str());
        return 1;
    }
    sf_writef_float(outFile, buf.data(), frames);
    sf_close(outFile);

    if (cfg.verbose) {
        fprintf(stderr, "[generate] Output written: %s\n",
                outputPath.c_str());
    }

    return 0;
}
