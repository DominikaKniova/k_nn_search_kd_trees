#pragma once

#include "boost/math/distributions/skew_normal.hpp"

#include <iostream>
#include <random>
#include <vector>
#include <queue>
#include <algorithm>
#include <ctime>
#include <chrono>
#include <utility>

#include "comparators.h"
#include "point.h"
#include "types.h"

//methods for generating datasets

class DataGenerator{
public:
	DataGenerator();

    void generate_uniform(unsigned int size, unsigned int dimension, std::vector<float> & ranges, std::vector<Point> & out_data);

    void generate_normal(unsigned int size, unsigned int dimension, std::vector<float> &means,
						 std::vector<float> &deviations, std::vector<Point> &out_data);

    void generate_skew_normal(unsigned int size, unsigned int dimension, std::vector<float> &locations,
							  std::vector<float> &scales, std::vector<float> &alphas, std::vector<Point> &out_data);

    void generate_circular(unsigned int size, unsigned int dimension, std::vector<float> &center,
						   float radius, std::vector<Point> &out_data);

    void generate_exponential(unsigned int size, unsigned int dimension, std::vector<float> & lambdas, std::vector<Point> & out_data);

};