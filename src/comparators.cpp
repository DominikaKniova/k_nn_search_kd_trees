#include "comparators.h"

bool kd_search_comparator::operator()(Point *lhs, Point *rhs) {
    return lhs->query_dist < rhs->query_dist;
}