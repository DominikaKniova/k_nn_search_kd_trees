#include "leaf_points.h"

Leaf_points::Leaf_points() {}

Leaf_points::Leaf_points(data_iterator & data_begin, data_iterator & data_end){
    this->data_begin = data_begin;
    this->data_end = data_end;
}