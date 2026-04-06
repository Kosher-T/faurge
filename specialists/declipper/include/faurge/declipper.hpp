// ═══════════════════════════════════════════════════════════════
// Faurge Specialist — De-Clipper: Main Public API
// ═══════════════════════════════════════════════════════════════
// This is the only header you need to include.
// Usage:
//     faurge::Declipper dc(config);
//     auto result = dc.processFile("in.wav", "out.wav");
//
// Or for in-memory processing:
//     std::vector<float> buf = ...;
//     auto result = dc.process(buf, sampleRate);
// ═══════════════════════════════════════════════════════════════
#pragma once

#include "clip_region.hpp"
#include <string>
#include <vector>

namespace faurge {

class Declipper {
public:
    explicit Declipper(const DeclipConfig& config = {});

    // Process audio in-place.  Returns full metrics.
    DeclipResult process(std::vector<float>& audio, int sampleRate);

    // Process a WAV file on disk.  Reads input, declips, writes output.
    DeclipResult processFile(const std::string& inputPath,
                             const std::string& outputPath);

    // Read-only access to the active config
    const DeclipConfig& config() const { return cfg_; }

private:
    DeclipConfig cfg_;
};

}  // namespace faurge
