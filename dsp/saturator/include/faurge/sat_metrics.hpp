#pragma once

#include "sat_types.hpp"
#include <string>

namespace faurge {

class SatMetrics {
public:
    static std::string toJson(const SatResult& result);
    static void printSummary(const SatResult& result);
};

} // namespace faurge
