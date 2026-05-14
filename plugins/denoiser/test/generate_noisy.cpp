#include <sndfile.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static constexpr float PI = 3.14159265358979f;

static void printUsage(const char* prog) {
    fprintf(stderr,
        "Faurge — Training Pair Generator (Denoiser)\n"
        "Creates noisy audio from clean audio for training data.\n\n"
        "Usage:\n"
        "  %s <clean.wav> <noisy.wav> [options]\n\n"
        "Options:\n"
        "  --noise-type <type>     Noise type: white|pink|babble  (default: white)\n"
        "  --snr-db <float>        Target SNR in dB              (default: 10.0)\n"
        "  --reverb                Add synthetic reverb\n"
        "  --reverb-rt60 <float>   Reverb RT60 in seconds         (default: 0.5)\n"
        "  --seed <int>            RNG seed                      (default: 42)\n"
        "  --help                  Print this message\n\n",
        prog);
}

static void addWhiteNoise(std::vector<float>& audio, float level,
                          std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& s : audio)
        s += dist(rng) * level;
}

static void addPinkNoise(std::vector<float>& audio, float level,
                         std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f, b4 = 0.0f, b5 = 0.0f;
    for (auto& s : audio) {
        float white = dist(rng);
        b0 = 0.99886f * b0 + white * 0.0555179f;
        b1 = 0.99332f * b1 + white * 0.0750759f;
        b2 = 0.96900f * b2 + white * 0.1538520f;
        b3 = 0.86650f * b3 + white * 0.3104856f;
        b4 = 0.55000f * b4 + white * 0.5329522f;
        b5 = -0.7616f * b5 - white * 0.0168980f;
        float pink = (b0 + b1 + b2 + b3 + b4 + b5 + white * 0.5362f) * 0.11f;
        s += pink * level;
    }
}

static void addBabbleNoise(std::vector<float>& audio, float level,
                           std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> modBuffer(audio.size());
    for (auto& s : modBuffer)
        s = dist(rng) * level;

    for (size_t i = 4; i < modBuffer.size(); ++i)
        modBuffer[i] = (modBuffer[i] + modBuffer[i-1] + modBuffer[i-2] + modBuffer[i-3] + modBuffer[i-4]) * 0.2f;

    for (size_t i = 0; i < audio.size(); ++i)
        audio[i] += modBuffer[i];
}

static void applyReverb(std::vector<float>& audio, float rt60,
                        int sampleRate) {
    int delayLen = static_cast<int>(rt60 * sampleRate);
    if (delayLen < 10) return;

    float decay = std::pow(0.001f, 1.0f / delayLen);
    std::vector<float> wet(audio.size(), 0.0f);

    for (size_t i = 0; i < audio.size(); ++i) {
        wet[i] = audio[i];
        if (i >= static_cast<size_t>(delayLen)) {
            wet[i] += wet[i - delayLen] * decay;
        }
    }

    for (size_t i = 0; i < audio.size(); ++i)
        audio[i] = 0.7f * audio[i] + 0.3f * wet[i];
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

    std::string cleanPath = argv[1];
    std::string noisyPath = argv[2];

    std::string noiseType = "white";
    float snrDb = 10.0f;
    bool useReverb = false;
    float reverbRt60 = 0.5f;
    int seed = 42;

    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--noise-type") == 0 && i + 1 < argc) {
            noiseType = argv[++i];
        } else if (std::strcmp(argv[i], "--snr-db") == 0 && i + 1 < argc) {
            snrDb = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--reverb") == 0) {
            useReverb = true;
        } else if (std::strcmp(argv[i], "--reverb-rt60") == 0 && i + 1 < argc) {
            reverbRt60 = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = std::atoi(argv[++i]);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

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

    std::mt19937 rng(seed);

    double signalPower = 0.0;
    for (const auto& s : audio)
        signalPower += static_cast<double>(s) * s;
    signalPower /= totalSamples;

    double noiseLevel = std::sqrt(signalPower / std::pow(10.0, snrDb / 10.0));

    if (noiseType == "pink") {
        addPinkNoise(audio, static_cast<float>(noiseLevel), rng);
    } else if (noiseType == "babble") {
        addBabbleNoise(audio, static_cast<float>(noiseLevel), rng);
    } else {
        addWhiteNoise(audio, static_cast<float>(noiseLevel), rng);
    }

    if (useReverb) {
        applyReverb(audio, reverbRt60, sfInfo.samplerate);
    }

    SF_INFO outInfo = sfInfo;
    outInfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    SNDFILE* outFile = sf_open(noisyPath.c_str(), SFM_WRITE, &outInfo);
    if (!outFile) {
        fprintf(stderr, "Error: Cannot write %s: %s\n",
                noisyPath.c_str(), sf_strerror(nullptr));
        return 1;
    }

    sf_writef_float(outFile, audio.data(), sfInfo.frames);
    sf_close(outFile);

    fprintf(stderr, "Generated noisy audio:\n");
    fprintf(stderr, "  Input:       %s\n", cleanPath.c_str());
    fprintf(stderr, "  Output:      %s\n", noisyPath.c_str());
    fprintf(stderr, "  Noise type:  %s\n", noiseType.c_str());
    fprintf(stderr, "  Target SNR:  %.1f dB\n", snrDb);
    if (useReverb)
        fprintf(stderr, "  Reverb:      RT60=%.2f s\n", reverbRt60);

    return 0;
}
