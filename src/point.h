#pragma once
#include <vector>

struct Point {
    std::vector<float> p;
    // based on point's query distance points are compared in priority queue
    float query_dist = 0;
    bool operator<(const Point & rhs) const;
};
