#pragma once

#include "transient_types.hpp"
#include <string>

namespace faurge {

class TransientMetrics {
public:
    static std::string toJson(const TransientResult& result);
    static void printSummary(const TransientResult& result);
};

} // namespace faurge
