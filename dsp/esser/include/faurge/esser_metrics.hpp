#pragma once

#include "esser_types.hpp"
#include <string>

namespace faurge {

class EsserMetrics {
public:
    static std::string toJson(const EsserResult& result);
    static void printSummary(const EsserResult& result);
};

} // namespace faurge
