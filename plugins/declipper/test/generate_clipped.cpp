// ═══════════════════════════════════════════════════════════════
// Faurge — Training Pair Generator
// ═══════════════════════════════════════════════════════════════
// Creates clipped audio from clean audio for training data.
//
// Usage:
//   faurge-generate-clipped clean.wav clipped.wav [options]
//
// Options:
//   --clip-db <float>       Gain in dB before hard-clipping (default: 6.0)
//   --soft-clip             Use tanh soft-clipping instead of hard-clip
//   --random-segments       Only clip random portions of the file
//   --seed <int>            RNG seed for reproducibility (default: 42)
//   --segment-chance <float> Probability of clipping each segment (default: 0.3)
// ═══════════════════════════════════════════════════════════════
#include <sndfile.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static void printUsage(const char* prog) {
    fprintf(stderr,
        "Faurge — Training Pair Generator\n"
        "Creates clipped audio from clean audio for cloud training.\n\n"
        "Usage:\n"
        "  %s <clean.wav> <clipped.wav> [options]\n\n"
        "Options:\n"
        "  --clip-db <float>          Gain before clipping in dB  (default: 6.0)\n"
        "  --soft-clip                Use tanh soft-clip           (default: hard)\n"
        "  --random-segments          Clip random segments only\n"
        "  --seed <int>               RNG seed                    (default: 42)\n"
        "  --segment-chance <float>   Clip probability per segment (default: 0.3)\n"
        "  --segment-size <int>       Segment size in samples     (default: 1024)\n"
        "  --help                     Print this message\n\n",
        prog);
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

    std::string cleanPath   = argv[1];
    std::string clippedPath = argv[2];

    float  clipDb          = 6.0f;
    bool   useSoftClip     = false;
    bool   randomSegments  = false;
    int    seed            = 42;
    float  segmentChance   = 0.3f;
    int    segmentSize     = 1024;

    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--clip-db") == 0 && i + 1 < argc) {
            clipDb = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--soft-clip") == 0) {
            useSoftClip = true;
        } else if (std::strcmp(argv[i], "--random-segments") == 0) {
            randomSegments = true;
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--segment-chance") == 0 && i + 1 < argc) {
            segmentChance = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--segment-size") == 0 && i + 1 < argc) {
            segmentSize = std::atoi(argv[++i]);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    // Read clean audio
    SF_INFO sfInfo;
    std::memset(&sfInfo, 0, sizeof(sfInfo));
    SNDFILE* inFile = sf_open(cleanPath.c_str(), SFM_READ, &sfInfo);
    if (!inFile) {
        fprintf(stderr, "Error: Cannot open %s: %s\n",
                cleanPath.c_str(), sf_strerror(nullptr));
        return 1;
    }

    size_t totalSamples = static_cast<size_t>(sfInfo.frames) * sfInfo.channels;
    std::vector<float> audio(totalSamples);
    sf_readf_float(inFile, audio.data(), sfInfo.frames);
    sf_close(inFile);

    // Apply gain
    float gainLinear = std::pow(10.0f, clipDb / 20.0f);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    size_t clippedCount = 0;

    if (randomSegments) {
        // Only clip random segments
        for (size_t i = 0; i < totalSamples; i += segmentSize) {
            bool shouldClip = dist(rng) < segmentChance;
            size_t end = std::min(i + static_cast<size_t>(segmentSize), totalSamples);

            if (shouldClip) {
                for (size_t j = i; j < end; ++j) {
                    float sample = audio[j] * gainLinear;
                    if (useSoftClip) {
                        sample = std::tanh(sample);
                    } else {
                        if (sample >  1.0f) { sample =  1.0f; ++clippedCount; }
                        if (sample < -1.0f) { sample = -1.0f; ++clippedCount; }
                    }
                    audio[j] = sample;
                }
            }
            // else: leave segment untouched
        }
    } else {
        // Clip entire file
        for (size_t i = 0; i < totalSamples; ++i) {
            float sample = audio[i] * gainLinear;
            if (useSoftClip) {
                sample = std::tanh(sample);
            } else {
                if (sample >  1.0f) { sample =  1.0f; ++clippedCount; }
                if (sample < -1.0f) { sample = -1.0f; ++clippedCount; }
            }
            audio[i] = sample;
        }
    }

    // Write clipped audio
    SF_INFO outInfo = sfInfo;
    outInfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    SNDFILE* outFile = sf_open(clippedPath.c_str(), SFM_WRITE, &outInfo);
    if (!outFile) {
        fprintf(stderr, "Error: Cannot write %s: %s\n",
                clippedPath.c_str(), sf_strerror(nullptr));
        return 1;
    }

    sf_writef_float(outFile, audio.data(), sfInfo.frames);
    sf_close(outFile);

    fprintf(stderr, "Generated clipped audio:\n");
    fprintf(stderr, "  Input:     %s\n", cleanPath.c_str());
    fprintf(stderr, "  Output:    %s\n", clippedPath.c_str());
    fprintf(stderr, "  Gain:      +%.1f dB (%.2fx)\n", clipDb, gainLinear);
    fprintf(stderr, "  Mode:      %s\n", useSoftClip ? "tanh soft-clip" : "hard-clip");
    fprintf(stderr, "  Segments:  %s\n", randomSegments ? "random" : "full file");
    if (!useSoftClip) {
        float pct = 100.0f * static_cast<float>(clippedCount)
                    / static_cast<float>(totalSamples);
        fprintf(stderr, "  Clipped:   %zu samples (%.2f%%)\n", clippedCount, pct);
    }

    return 0;
}
