#ifndef PROJECT_RANGE_SEARCH_STACK_STR_H
#define PROJECT_RANGE_SEARCH_STACK_STR_H

#include "kd_node.h"

// structures used in kd tree search algorithms

struct kNN_search_str{
    kNN_search_str();
    kNN_search_str(kd_node * node, std::vector<float> & bounds, float rd);
    kd_node * node;
    std::vector<float> bounds;
    float rd;
    bool operator<(const kNN_search_str & rhs) const;
    bool operator>(const kNN_search_str & rhs) const;
};

struct kNN_search_str_vis{
    kNN_search_str_vis();
    kNN_search_str_vis(kd_node * node, std::vector<float> & bounds, float rd);
    kd_node * node;
    std::vector<float> bounds;
    float rd;
    bool operator<(const kNN_search_str_vis & rhs) const;
    bool operator>(const kNN_search_str_vis & rhs) const;
};

struct Range_search_sph{
    Range_search_sph();
    Range_search_sph(kd_node * node, std::vector<float> & off, float rd);
    kd_node * node;
    std::vector<float> off;
    float rd;
};

struct Range_search_sph_vis{
    Range_search_sph_vis();
    Range_search_sph_vis(kd_node * node, std::vector<float> & off, float rd, std::vector<float> & bounds);
    kd_node * node;
    std::vector<float> off;
    float rd;
    std::vector<float> bounds;
};

struct Range_search_rect_vis{
    Range_search_rect_vis();
    Range_search_rect_vis(kd_node * node, std::vector<float> & bounds);
    kd_node * node;
    std::vector<float> bounds;
};

#endif //PROJECT_RANGE_SEARCH_STACK_STR_H
