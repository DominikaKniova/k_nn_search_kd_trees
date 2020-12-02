#include <vector>
#include <queue>
#include <stack>
#include <climits>
#include <cmath>
#include <algorithm>
#include <set>

#include "comparators.h"
#include "data.h"
#include "data_generator.h"
#include "kd_node.h"
#include "search_structures.h"
#include "types.h"
#include "utils.h"

#include "binomial_heap.h"

#define MIN 0
#define MAX 1
#define LEAF_NODE 4

class kd_tree {
public:
    kd_node root;

    unsigned int dimensionality = 0;

	unsigned int max_points_in_leaf = 1;

	unsigned int vis_leaves = 0;

	bool visualization_mode = false;

	kd_node * tree_nodes;

	Leaf_points * leaf_data;

	std::vector<float> bounding_box;

	std::deque<std::vector<float>> splitting_lines;

	// arrays for visualizations
	std::vector<std::vector<float>> kNN_visited_leaves;

	std::vector<std::vector<float>> range_search_sph_visited_leaves;

	std::vector<std::vector<float>> range_search_rect_visited_leaves;

	kd_tree();

	kd_tree(unsigned int dimensionality, unsigned int max_points_in_leaf, bool visualization_mode);

    void build(std::vector<Point> & data);

    void get_bounding_box(std::vector<Point> & data);

    // =================================================================================================================
    // search methods:
    void k_NN_search_priority_queue(const int &k, const VectorXf &query, std::deque<Point *> &out_data);

	void k_NN_search_binomial_heap(const int &k, const VectorXf &query, std::deque<Point *> &out_data);

    void k_NN_search_priority_queue_vis(const int &k, const VectorXf &query, std::deque<Point *> &out_data);

    void k_NN_search_binomial_heap_vis(const int &k, const VectorXf &query, std::deque<Point *> &out_data);

    void range_search_rectangular(const VectorXf & query, const std::vector<float> & range_sizes, std::deque<Point *> & out_data);

	void range_search_rectangular_vis(const VectorXf & query, const std::vector<float> & range_sizes, std::deque<Point *> & out_data);

    void range_search_spherical(const VectorXf & query, const float & rad, std::deque<Point *> & out_data);

    void range_search_spherical_vis(const VectorXf & query, const float & rad, std::deque<Point *> & out_data);

   // processing leaf methods
    void process_leaf_kNN(kd_node * node, std::priority_queue<Point *, std::vector<Point *>, kd_search_comparator> & kNN_q, const VectorXf & query, const int & k, float & kNN_dist);

    void process_leaf_kNN_vis(kd_node *node,
                              std::priority_queue<Point *, std::vector<Point *>, kd_search_comparator> &kNN_q,
                              const VectorXf &query, const int &k, float &kNN_dist, std::vector<float> &bounds);

    void process_leaf_rect_range_search(kd_node * node, const VectorXf & query, const std::vector<float> & range_sizes, std::deque<Point *> & in_range);

	void process_leaf_rect_range_search_vis(kd_node * node, const VectorXf & query, const std::vector<float> & range_sizes, std::deque<Point *> & in_range, std::vector<float> & bounds);

    void process_leaf_sph_range_search(kd_node * node, const VectorXf & query, float & rad_sqr, std::deque<Point *> & in_range);

	void process_leaf_sph_range_search_vis(kd_node * node, const VectorXf & query, float & rad_sqr, std::deque<Point *> & in_range, std::vector<float> & bounds);

    // =================================================================================================================

    void destroy_tree();

    // method for storing data for later visualizatoin (only in visualization mode)
    void store_vis_splitting_lines(int split_dim, float split_value, std::vector<float> & bounds);

private:
	// recursive kd tree build algorithm
    kd_node build_recursively(data_iterator data_begin, data_iterator data_end, std::vector<float> & bounds, int & curr_idx_nodes, int & curr_idx_leaves);

	// methods for getting splitting dimension, splitting value and partitioning data
	int get_most_extent_dimension(std::vector<float> & bounds);

	float get_splitting_value(std::vector<float> &bounds, int in_dim);

    data_iterator sliding_midpoint_partitioning(int split_dim, float &split_value, data_iterator data_begin,
												data_iterator data_end);
};