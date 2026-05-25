#pragma once

#include "gain_types.hpp"
#include <string>

namespace faurge {

class GainMetrics {
public:
    static std::string toJson(const GainResult& result);
    static void printSummary(const GainResult& result);
};

} // namespace faurge
