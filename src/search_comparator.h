#ifndef PROJECT_SEARCH_COMPARATOR_H
#define PROJECT_SEARCH_COMPARATOR_H

#include <vector>
#include <string>
#include <fstream>
#include "kd_tree.h"
#include "naive_search.h"
#include "point.h"
#include "types.h"
#include "visualizer.h"

class SearchComparator{
private:
    int dimensionality;
    std::vector<Point> * data;
    kd_tree * kdtree;

    // color definitions for visualization
    double black[3] = {0.0, 0.0, 0.0};
    double red  [3] = {1.0, 0.0, 0.0};
    double green[3] = {0.0, 1.0, 0.0};
    double blue [3] = {0.0, 0.0, 1.0};

public:
    SearchComparator();

    SearchComparator(int dimensionality, kd_tree * kd_tree, std::vector<Point> & data);

    void compare_kNN_search(const int k, const VectorXf & query, bool pq, bool vi);

    void compare_range_search_spherical(const float radius, const VectorXf query, bool vi);

    void compare_range_search_rectangular(std::vector<float> & ranges, const VectorXf query, bool vi);

    static void experiment_1(unsigned int num_samples, unsigned int dimensionality, int max_points_in_leaf, int k);

    static void experiment_1_build_diff_distributions_fix_dim_mlp_var_N_dparams(unsigned int dim, unsigned  int max_leaf_point);

    static void experiment_2_build_diff_distr_fix_N_dim_var_mlp(unsigned int num_samples, unsigned int dim);

    static void experiment_3_build_diff_distr_fix_mlp_dim_var_N(unsigned int dim, unsigned  int max_leaf_point);

    static void experiment_4_kNN_fix_N_dim_k_var_mlp(unsigned int num_samples, unsigned int dim,
                                                     std::vector<int> & ks, std::vector<float> & radii, std::vector<std::vector<float>> & rect_range, bool binom_heap);
};

#endif //PROJECT_SEARCH_COMPARATOR_H
