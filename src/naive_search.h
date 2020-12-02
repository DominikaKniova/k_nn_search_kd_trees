#ifndef PROJECT_NAIVE_SEARCH_H
#define PROJECT_NAIVE_SEARCH_H

#include <queue>
#include <vector>
#include <climits>

#include "comparators.h"
#include "types.h"
#include "utils.h"

void naive_k_NN(const int & k, const VectorXf & query, std::vector<Point> & data, std::deque<Point *> & out_data);

void naive_range_search_spherical(const VectorXf &query, const float rad, std::vector<Point> & data, std::deque<Point *> & out_data);

void naive_range_search_rectangular(const VectorXf &query, const std::vector<float> &range_sizes, const int dimensionality, std::vector<Point> & data, std::deque<Point *> & out_data);

#endif //PROJECT_NAIVE_SEARCH_H
