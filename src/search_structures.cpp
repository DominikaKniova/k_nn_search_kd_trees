#include "search_structures.h"

// structures used in kd tree search algorithms

kNN_search_str::kNN_search_str() = default;

kNN_search_str::kNN_search_str(kd_node *node, std::vector<float> &bounds, float rd) {
    this->node = node;
    this->bounds = bounds;
    this->rd = rd;
}

bool kNN_search_str::operator<(const kNN_search_str &rhs) const {
    return this->rd > rhs.rd;
}

bool kNN_search_str::operator>(const kNN_search_str &rhs) const {
    return this->rd < rhs.rd;
}

kNN_search_str_vis::kNN_search_str_vis() = default;

kNN_search_str_vis::kNN_search_str_vis(kd_node *node, std::vector<float> &bounds, float rd) {
    this->node = node;
    this->bounds = bounds;
    this->rd = rd;
}

bool kNN_search_str_vis::operator<(const kNN_search_str_vis &rhs) const {
    return this->rd > rhs.rd;
}

bool kNN_search_str_vis::operator>(const kNN_search_str_vis &rhs) const {
    return this->rd < rhs.rd;
}

Range_search_sph::Range_search_sph() = default;

Range_search_sph::Range_search_sph(kd_node *node, std::vector<float> & off, float rd) {
    this->node = node;
    this->off = std::move(off);
    this->rd = rd;
}

Range_search_sph_vis::Range_search_sph_vis() = default;

Range_search_sph_vis::Range_search_sph_vis(kd_node *node, std::vector<float> &off, float rd,
                                           std::vector<float> & bounds) {
    this->node = node;
    this->off = std::move(off);
    this->rd = rd;
    this->bounds = bounds;
}

Range_search_rect_vis::Range_search_rect_vis() = default;

Range_search_rect_vis::Range_search_rect_vis(kd_node *node, std::vector<float> &bounds) {
    this->node = node;
    this->bounds = bounds;
}