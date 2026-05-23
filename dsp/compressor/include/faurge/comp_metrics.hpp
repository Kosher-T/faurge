#pragma once

#include "comp_types.hpp"
#include <string>

namespace faurge {

class CompMetrics {
public:
    static std::string toJson(const CompResult& result);
    static void printSummary(const CompResult& result);
};

}
