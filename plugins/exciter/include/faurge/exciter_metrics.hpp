#pragma once

#include "exciter_types.hpp"
#include <string>

namespace faurge {

class ExciterMetrics {
public:
    static std::string toJson(const ExciterResult& result);
    static void printSummary(const ExciterResult& result);
};

}
