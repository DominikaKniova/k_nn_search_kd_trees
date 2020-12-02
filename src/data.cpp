#include "data.h"

Data::Data(const data_iterator data_begin, const data_iterator data_end, const std::vector<float> & bounds) {
    this->data_begin = data_begin;
    this->data_end = data_end;
    this->bounds = bounds;
}
