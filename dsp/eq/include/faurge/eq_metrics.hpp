#pragma once

#include "eq_types.hpp"
#include <string>

namespace faurge {

class EqMetrics {
public:
    static std::string toJson(const EqResult& result);
    static void printSummary(const EqResult& result);
};

}
