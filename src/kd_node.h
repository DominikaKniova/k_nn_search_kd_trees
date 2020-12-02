#pragma once

#include <vector>
#include <stdint.h>

#include "leaf_points.h"
#include "point.h"
#include "types.h"

struct kd_node {
	int index;  // index to left child in array of nodes (internal node)
				// right child at index
				// or index to array of data info (leaf node) storing iterator to first and last point of this leaf
    int is_leaf_split_dim = 0;
    float split_value;
};