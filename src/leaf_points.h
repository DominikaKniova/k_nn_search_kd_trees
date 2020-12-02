#ifndef PROJECT_LEAF_POINTS_H
#define PROJECT_LEAF_POINTS_H

#include "types.h"

// structure storing information about data contained in a leaf
struct Leaf_points{
    Leaf_points();
    Leaf_points(data_iterator & data_begin, data_iterator & data_end);
    // points in vector in range [data_begin, data_end) are contained in leaf
    data_iterator data_begin;
    data_iterator data_end;
};

#endif //PROJECT_LEAF_POINTS_H
