#pragma once

#include "lim_types.hpp"
#include <string>

namespace faurge {

class LimiterMetrics {
public:
    static std::string toJson(const LimiterResult& result);
    static void printSummary(const LimiterResult& result);
};

} // namespace faurge
