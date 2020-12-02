#ifndef PROJECT_DATA_H
#define PROJECT_DATA_H

#include "types.h"

struct Data{
    Data(const data_iterator data_begin, const data_iterator data_end, const std::vector<float> & bounds);
    data_iterator data_begin;
    data_iterator data_end;
    std::vector<float> bounds;
};

#endif //PROJECT_DATA_H
