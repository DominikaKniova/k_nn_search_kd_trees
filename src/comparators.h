#ifndef PROJECT_COMPARATORS_H
#define PROJECT_COMPARATORS_H

#include "kd_node.h"
#include "types.h"
#include "point.h"

struct kd_search_comparator {
    bool operator () (Point * lhs, Point* rhs);
    bool operator () (kd_node * lhs, kd_node * rhs);
    bool operator () (search_structure & lhs, search_structure & rhs);
};

#endif //PROJECT_COMPARATORS_H
