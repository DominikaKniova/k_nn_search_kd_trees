#include "point.h"

bool Point::operator<(const Point &rhs) const {
    return query_dist < rhs.query_dist;
}
