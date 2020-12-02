#include "kd_tree.h"

kd_tree::kd_tree() {}

kd_tree::kd_tree(unsigned int dimensionality, unsigned int max_points_in_leaf, bool visualization_mode) {
    this->dimensionality = dimensionality;
    this->max_points_in_leaf = max_points_in_leaf;
    this->visualization_mode = visualization_mode;
}

kd_node kd_tree::build_recursively(data_iterator data_begin, data_iterator data_end, std::vector<float> & bounds, int & curr_idx_nodes, int & curr_idx_leaves) {
    kd_node node;

    // if there is less data to be divided than max_points_in_leaf -> create leaf and do not divide
    if (std::distance(data_begin, data_end) <= this->max_points_in_leaf){
        this->leaf_data[curr_idx_leaves].data_begin = data_begin;
        this->leaf_data[curr_idx_leaves].data_end = data_end;
        node.index = curr_idx_leaves;
        node.is_leaf_split_dim = LEAF_NODE;
        curr_idx_leaves++;
        return node;
    }

    // get splitting dimension and value (middle value of most extent dimension)
    node.is_leaf_split_dim = get_most_extent_dimension(bounds);
    node.split_value = get_splitting_value(bounds, node.is_leaf_split_dim);

    // partition data using sliding midpoint method and store partitioning iterator
    auto split_iterator = sliding_midpoint_partitioning(node.is_leaf_split_dim, node.split_value, data_begin, data_end);

    if (this->visualization_mode){
        store_vis_splitting_lines(node.is_leaf_split_dim, node.split_value, bounds);
    }

    // updates bounds for left subtree
    float old_bound = bounds[node.is_leaf_split_dim * 2 + MAX];
    bounds[node.is_leaf_split_dim * 2 + MAX] = node.split_value;

    // recursively build left subtree
    kd_node node_left = build_recursively(data_begin, split_iterator, bounds, curr_idx_nodes, curr_idx_leaves);

    // updates bounds for right subtree
    bounds[node.is_leaf_split_dim * 2 + MAX] = old_bound;
    old_bound = bounds[node.is_leaf_split_dim * 2 + MIN];
    bounds[node.is_leaf_split_dim * 2 + MIN] = node.split_value;

    // recursively build right subtree
    kd_node node_right = build_recursively(split_iterator, data_end, bounds, curr_idx_nodes, curr_idx_leaves);

    // get bounds to previous state
    bounds[node.is_leaf_split_dim * 2 + MIN] = old_bound;

    // store children in tree_nodes array in correct order
    this->tree_nodes[curr_idx_nodes] = node_left;
    // store index of left child in the node. Right child is stored is stored on index + 1
    node.index = curr_idx_nodes;
    curr_idx_nodes++;
    this->tree_nodes[curr_idx_nodes] = node_right;
    curr_idx_nodes++;

    return node;
}

void kd_tree::build(std::vector<Point> & data) {
    // reserve enough memory for tree nodes and leaf data
    this->tree_nodes = new kd_node[2 * data.size() - 1];
    this->leaf_data = new Leaf_points[data.size()];

    get_bounding_box(data);

    std::vector<float> bounds = this->bounding_box;
    // indices to array of tree nodes and leaf data
    int idx_nodes = 0;
    int idx_leaves = 0;
    this->root = build_recursively(data.begin(), data.end(), bounds, idx_nodes, idx_leaves);
}

void kd_tree::get_bounding_box(std::vector<Point> & data){
    // initialize array
    this->bounding_box.reserve(2 * this->dimensionality);
    for (int k = 0; k < this->dimensionality; ++k) {
        this->bounding_box.push_back(std::numeric_limits<float>::max());
        this->bounding_box.push_back(-std::numeric_limits<float>::max());
    }
    // find minimum and maximum bounds in every dimension
    for (unsigned int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < this->dimensionality; ++j) {
            if (data[i].p[j] > this->bounding_box[j* 2 + MAX]){
                this->bounding_box[j * 2 + MAX] = data[i].p[j];
            }

            if (data[i].p[j] < this->bounding_box[j* 2 + MIN]){
                this->bounding_box[j* 2 + MIN] = data[i].p[j];
            }
        }
    }
}

void kd_tree::k_NN_search_priority_queue(const int &k, const VectorXf &query, std::deque<Point *> &out_data){
    this->vis_leaves = 0;
    float kNN_dist = std::numeric_limits<float>::max();

    // priority queue storing k closest neighbours. Farthest neighbour is first in the priority queue (due to getting kNN_dist)
    std::priority_queue<Point *, std::vector<Point *>, kd_search_comparator> kNN_q;
    // search priority queue
    std::priority_queue<kNN_search_str> q;

    // initialize queue with root node
    q.push(kNN_search_str(&this->root, this->bounding_box, 0.0f));

    // temporary variables for search
    int idx_child;
    kNN_search_str str;
    float offset_split_plane;
    float offset_box_boundary;
    float dist_further_child;
    float old_bound;

    while (!q.empty()){
        str = q.top();
        q.pop();

        if (str.rd >= kNN_dist && kNN_q.size() >= k){
            // we have k nearest neigbours and we know that in queue there are only nodes with bigger distance than kNN_dist -> we can end
            break;
        }

        if (str.rd < kNN_dist){
            while (str.node->is_leaf_split_dim != LEAF_NODE){

                // based on sign of offset between split plane and query decide which subtree should be searched first
                offset_split_plane = query[str.node->is_leaf_split_dim] - str.node->split_value;

                if (offset_split_plane <= 0.0f){
                    // left subtree will be searched first
                    offset_box_boundary = query[str.node->is_leaf_split_dim] - str.bounds[str.node->is_leaf_split_dim * 2 + MIN];

                    if (offset_box_boundary > 0.0f){
                        // query is inside -> set to zero
                        offset_box_boundary = 0.0f;
                    }

                    // compute distance to right child
                    dist_further_child = str.rd - offset_box_boundary * offset_box_boundary + offset_split_plane * offset_split_plane;

                    // update boundaries for right child
                    old_bound = str.bounds[str.node->is_leaf_split_dim * 2 + MIN];
                    str.bounds[str.node->is_leaf_split_dim * 2 + MIN] = str.node->split_value;

                    // push right node to priority queue with distance to this child
                    idx_child = str.node->index;
                    q.push(kNN_search_str(&this->tree_nodes[idx_child + 1], str.bounds, dist_further_child));

                    // restore boundaries for left child
                    str.bounds[str.node->is_leaf_split_dim * 2 + MIN] = old_bound;
                    str.bounds[str.node->is_leaf_split_dim * 2 + MAX] = str.node->split_value;

                    // search left subtree
                    str.node = &this->tree_nodes[idx_child];
                }
                else {
                    // right subtree will be searched first
                    offset_box_boundary = str.bounds[str.node->is_leaf_split_dim * 2 + MAX] - query[str.node->is_leaf_split_dim];

                    if (offset_box_boundary > 0.0f){
                        // query is inside -> set to zero
                        offset_box_boundary = 0.0f;
                    }

                    // compute distance to left child
                    dist_further_child = str.rd - offset_box_boundary * offset_box_boundary + offset_split_plane * offset_split_plane;

                    //update boundaries for left child
                    old_bound = str.bounds[str.node->is_leaf_split_dim * 2 + MAX];
                    str.bounds[str.node->is_leaf_split_dim * 2 + MAX] = str.node->split_value;

                    // push left node to priority queue with distance to this child
                    idx_child = str.node->index;
                    q.push(kNN_search_str(&this->tree_nodes[idx_child], str.bounds, dist_further_child));

                    //restore boundaries for right child
                    str.bounds[str.node->is_leaf_split_dim * 2 + MAX] = old_bound;
                    str.bounds[str.node->is_leaf_split_dim * 2 + MIN] = str.node->split_value;

                    // search right subtree
                    str.node = &this->tree_nodes[idx_child + 1];
                }
            }
            // update number of visited leaves for later statistics
            this->vis_leaves ++ ;
            process_leaf_kNN(str.node, kNN_q, query, k, kNN_dist);
        }
    }

    // push found k nearest neighbours to out array
    for (unsigned int i = 0; i < k; ++i) {
        out_data.push_back(kNN_q.top());
        kNN_q.pop();
    }
}

void kd_tree::k_NN_search_priority_queue_vis(const int &k, const VectorXf &query, std::deque<Point *> &out_data){
    this->vis_leaves = 0;
    float kNN_dist = std::numeric_limits<float>::max();

    // priority queue storing k closest neighbours. Farthest neighbour is first in the priority queue (due to getting kNN_dist)
    std::priority_queue<Point *, std::vector<Point *>, kd_search_comparator> kNN_q;
    // search priority queue
    std::priority_queue<kNN_search_str> q;

    // initialize queue with root node
    q.push(kNN_search_str(&this->root, this->bounding_box, 0.0f));

    // temporary variables for search
    int idx_child;
    kNN_search_str str;
    float offset_split_plane;
    float offset_box_boundary;
    float dist_further_child;
    float old_bound;

    while (!q.empty()){
        str = q.top();
        q.pop();

        if (str.rd >= kNN_dist && kNN_q.size() >= k){
            // we have k nearest neigbours and we know that in queue there are only nodes with bigger distance than kNN_dist -> we can end
            break;
        }

        if (str.rd < kNN_dist){
            while (str.node->is_leaf_split_dim != LEAF_NODE){

                // based on sign of offset between split plane and query decide which subtree should be searched first
                offset_split_plane = query[str.node->is_leaf_split_dim] - str.node->split_value;

                if (offset_split_plane <= 0.0f){
                    // left subtree will be searched first
                    offset_box_boundary = query[str.node->is_leaf_split_dim] - str.bounds[str.node->is_leaf_split_dim * 2 + MIN];

                    if (offset_box_boundary > 0.0f){
                        // query is inside
                        offset_box_boundary = 0.0f;
                    }

                    // compute distance to right child
                    dist_further_child = str.rd - offset_box_boundary * offset_box_boundary + offset_split_plane * offset_split_plane;

                    // update boundaries for right child
                    old_bound = str.bounds[str.node->is_leaf_split_dim * 2 + MIN];
                    str.bounds[str.node->is_leaf_split_dim * 2 + MIN] = str.node->split_value;

                    // push right node to priority queue with distance to this child
                    idx_child = str.node->index;
                    q.push(kNN_search_str(&this->tree_nodes[idx_child + 1], str.bounds, dist_further_child));

                    // restore boundaries for left child
                    str.bounds[str.node->is_leaf_split_dim * 2 + MIN] = old_bound;
                    str.bounds[str.node->is_leaf_split_dim * 2 + MAX] = str.node->split_value;

                    // search left subtree
                    str.node = &this->tree_nodes[idx_child];
                }
                else {
                    // right subtree will be searched first
                    offset_box_boundary = str.bounds[str.node->is_leaf_split_dim * 2 + MAX] - query[str.node->is_leaf_split_dim];

                    if (offset_box_boundary > 0.0f){
                        // query is inside -> set to zero
                        offset_box_boundary = 0.0f;
                    }

                    // compute distance to left child
                    dist_further_child = str.rd - offset_box_boundary * offset_box_boundary + offset_split_plane * offset_split_plane;

                    //update boundaries for left child
                    old_bound = str.bounds[str.node->is_leaf_split_dim * 2 + MAX];
                    str.bounds[str.node->is_leaf_split_dim * 2 + MAX] = str.node->split_value;

                    // push left node to priority queue with distance to this child
                    idx_child = str.node->index;
                    q.push(kNN_search_str(&this->tree_nodes[idx_child], str.bounds, dist_further_child));

                    //restore boundaries for right child
                    str.bounds[str.node->is_leaf_split_dim * 2 + MAX] = old_bound;
                    str.bounds[str.node->is_leaf_split_dim * 2 + MIN] = str.node->split_value;

                    // search right subtree
                    str.node = &this->tree_nodes[idx_child + 1];
                }
            }
            // update number of visited leaves for later statistics
            this->vis_leaves ++ ;
            process_leaf_kNN_vis(str.node, kNN_q, query, k, kNN_dist, str.bounds);
        }
    }
    // push found k nearest neighbours to out array
    for (unsigned int i = 0; i < k; ++i) {
        out_data.push_back(kNN_q.top());
        kNN_q.pop();
    }
}

void kd_tree::k_NN_search_binomial_heap(const int &k, const VectorXf &query, std::deque<Point *> &out_data) {
    this->vis_leaves = 0;
    float kNN_dist = std::numeric_limits<float>::max();

    // priority queue storing k closest neighbours. Farthest neighbour is first in the priority queue (due to getting kNN_dist)
    std::priority_queue<Point *, std::vector<Point *>, kd_search_comparator> kNN_q;
    // search priority queue (using binomial heap)
    BinomialHeap<kNN_search_str> fh;

    // initialize queue with root node
    fh.insert(kNN_search_str(&this->root, this->bounding_box, 0.0f));

    // temporary variables for search
    int idx_child;
    kNN_search_str str;
    float offset_split_plane;
    float offset_box_boundary;
    float dist_further_child;
    float old_bound;

    // search algorithm analogous to implementation with std::priority_queue
    while (!fh.is_empty()){
        str = fh.top();
        fh.pop();

        if (str.rd >= kNN_dist && kNN_q.size() >= k){
            // we have k nearest neigbours and we know that in queue there are only nodes with bigger distance than kNN_dist -> we can end
            break;
        }

        if (str.rd < kNN_dist){
            while (str.node->is_leaf_split_dim != LEAF_NODE){

                offset_split_plane = query[str.node->is_leaf_split_dim] - str.node->split_value;

                if (offset_split_plane <= 0.0f){
                    //left child
                    offset_box_boundary = query[str.node->is_leaf_split_dim] - str.bounds[str.node->is_leaf_split_dim * 2 + MIN];

                    if (offset_box_boundary > 0.0f){
                        offset_box_boundary = 0.0f;
                    }

                    dist_further_child = str.rd - offset_box_boundary * offset_box_boundary + offset_split_plane * offset_split_plane;
                    idx_child = str.node->index;

                    //update boundaries for right child
                    old_bound = str.bounds[str.node->is_leaf_split_dim * 2 + MIN];
                    str.bounds[str.node->is_leaf_split_dim * 2 + MIN] = str.node->split_value;

                    fh.insert(kNN_search_str(&this->tree_nodes[idx_child + 1], str.bounds, dist_further_child));

                    //update boundaries for left child
                    str.bounds[str.node->is_leaf_split_dim * 2 + MIN] = old_bound;
                    str.bounds[str.node->is_leaf_split_dim * 2 + MAX] = str.node->split_value;

                    str.node = &this->tree_nodes[idx_child];
                }
                else {
                    //right child
                    offset_box_boundary = str.bounds[str.node->is_leaf_split_dim * 2 + MAX] - query[str.node->is_leaf_split_dim];

                    if (offset_box_boundary > 0.0f){
                        offset_box_boundary = 0.0f;
                    }

                    dist_further_child = str.rd - offset_box_boundary * offset_box_boundary + offset_split_plane * offset_split_plane;
                    idx_child = str.node->index;

                    //update boundaries for left child
                    old_bound = str.bounds[str.node->is_leaf_split_dim * 2 + MAX];
                    str.bounds[str.node->is_leaf_split_dim * 2 + MAX] = str.node->split_value;

                    fh.insert(kNN_search_str(&this->tree_nodes[idx_child], str.bounds, dist_further_child));

                    //update boundaries for right child
                    str.bounds[str.node->is_leaf_split_dim * 2 + MAX] = old_bound;
                    str.bounds[str.node->is_leaf_split_dim * 2 + MIN] = str.node->split_value;

                    str.node = &this->tree_nodes[idx_child + 1];
                }
            }
            this->vis_leaves ++ ;
            process_leaf_kNN(str.node, kNN_q, query, k, kNN_dist);
        }
    }
    for (unsigned int i = 0; i < k; ++i) {
        out_data.push_back(kNN_q.top());
        kNN_q.pop();
    }

}

void kd_tree::k_NN_search_binomial_heap_vis(const int &k, const VectorXf &query, std::deque<Point *> &out_data) {
    this->vis_leaves = 0;
    float kNN_dist = std::numeric_limits<float>::max();

    // priority queue storing k closest neighbours. Farthest neighbour is first in the priority queue (due to getting kNN_dist)
    std::priority_queue<Point *, std::vector<Point *>, kd_search_comparator> kNN_q;
    // search priority queue (using binomial heap)
    BinomialHeap<kNN_search_str> fh;

    // initialize queue with root node
    fh.insert(kNN_search_str(&this->root, this->bounding_box, 0.0f));

    // temporary variables for search
    int idx_child;
    kNN_search_str str;
    float offset_split_plane;
    float offset_box_boundary;
    float dist_further_child;
    float old_bound;

    while (!fh.is_empty()){
        str = fh.top();
        fh.pop();

        if (str.rd >= kNN_dist && kNN_q.size() >= k){
            // we have k nearest neigbours and we know that in queue there are only nodes with bigger distance than kNN_dist -> we can end
            break;
        }

        if (str.rd < kNN_dist){
            while (str.node->is_leaf_split_dim != LEAF_NODE){

                offset_split_plane = query[str.node->is_leaf_split_dim] - str.node->split_value;

                if (offset_split_plane <= 0.0f){
                    //left child
                    offset_box_boundary = query[str.node->is_leaf_split_dim] - str.bounds[str.node->is_leaf_split_dim * 2 + MIN];

                    if (offset_box_boundary > 0.0f){
                        offset_box_boundary = 0.0f;
                    }

                    dist_further_child = str.rd - offset_box_boundary * offset_box_boundary + offset_split_plane * offset_split_plane;
                    idx_child = str.node->index;


                    //update boundaries for right child
                    old_bound = str.bounds[str.node->is_leaf_split_dim * 2 + MIN];
                    str.bounds[str.node->is_leaf_split_dim * 2 + MIN] = str.node->split_value;

                    fh.insert(kNN_search_str(&this->tree_nodes[idx_child + 1], str.bounds, dist_further_child));

                    //update boundaries for left child
                    str.bounds[str.node->is_leaf_split_dim * 2 + MIN] = old_bound;
                    str.bounds[str.node->is_leaf_split_dim * 2 + MAX] = str.node->split_value;

                    str.node = &this->tree_nodes[idx_child];
                }
                else {
                    //right child
                    offset_box_boundary = str.bounds[str.node->is_leaf_split_dim * 2 + MAX] - query[str.node->is_leaf_split_dim];

                    if (offset_box_boundary > 0.0f){
                        offset_box_boundary = 0.0f;
                    }

                    dist_further_child = str.rd - offset_box_boundary * offset_box_boundary + offset_split_plane * offset_split_plane;
                    idx_child = str.node->index;

                    //update boundaries for left child
                    old_bound = str.bounds[str.node->is_leaf_split_dim * 2 + MAX];
                    str.bounds[str.node->is_leaf_split_dim * 2 + MAX] = str.node->split_value;

                    fh.insert(kNN_search_str(&this->tree_nodes[idx_child], str.bounds, dist_further_child));

                    //update boundaries for right child
                    str.bounds[str.node->is_leaf_split_dim * 2 + MAX] = old_bound;
                    str.bounds[str.node->is_leaf_split_dim * 2 + MIN] = str.node->split_value;

                    str.node = &this->tree_nodes[idx_child + 1];
                }
            }
            this->vis_leaves ++ ;
            process_leaf_kNN_vis(str.node, kNN_q, query, k, kNN_dist, str.bounds);
        }
    }
    for (unsigned int i = 0; i < k; ++i) {
        out_data.push_back(kNN_q.top());
        kNN_q.pop();
    }

}

void kd_tree::process_leaf_kNN(kd_node * node, std::priority_queue<Point *, std::vector<Point *>, kd_search_comparator> & kNN_q, const VectorXf & query, const int & k, float & kNN_dist){
    // temp distance variable
    float d;

    // get leaf data information
    Leaf_points lp = this->leaf_data[node->index] ;

    for (auto it = lp.data_begin; it != lp.data_end; it++) {
        if ((*it).p == query){
            continue;
        }

        d = dist_sqr((*it).p, query);
        // update point's query distance so it can be inserted to the correct position in prior. queue
        (*it).query_dist = d;

        // first in kNN are the farthest points from query, last is closest
        if (kNN_q.size() < k){
            kNN_q.push(&(*it));
        }
        else if (d < kNN_dist){
            kNN_q.push(&(*it));
            if (kNN_q.size() > k){
                kNN_q.pop();
                // because first in queue is farthest, we can easily get current kNN distance
                kNN_dist = kNN_q.top()->query_dist;
            }
        }
    }
}

void kd_tree::process_leaf_kNN_vis(kd_node *node,
                                   std::priority_queue<Point *, std::vector<Point *>, kd_search_comparator> &kNN_q,
                                   const VectorXf &query, const int &k, float &kNN_dist, std::vector<float> &bounds){

    // the same algorithm as in process_leaf_kNN but also storing the bounds of visited leaves (for later visualization)
    if (this->visualization_mode){
        this->kNN_visited_leaves.push_back(bounds);
    }

    float d;
    Leaf_points lp = this->leaf_data[node->index];

    for (auto it = lp.data_begin; it != lp.data_end; it++) {
        if ((*it).p == query){
            continue;
        }

        d = dist_sqr((*it).p, query);
        (*it).query_dist = d;

        // first in kNN are the farthest points from query, last is closest
        if (kNN_q.size() < k){
            kNN_q.push(&(*it));
        }
        else if (d < kNN_dist){
            kNN_q.push(&(*it));
            if (kNN_q.size() > k){
                kNN_q.pop();
                kNN_dist = kNN_q.top()->query_dist;
            }
        }
    }
}

void kd_tree::range_search_rectangular(const VectorXf & query, const std::vector<float> & range_sizes, std::deque<Point *> & out_data) {
    this->vis_leaves = 0;
    std::stack<kd_node *> s;
    int idx_child;
    s.push(&this->root);
    kd_node * node;

    while (!s.empty()) {
        node = s.top();
        s.pop();

        if (node->is_leaf_split_dim == LEAF_NODE){
            vis_leaves ++;
            process_leaf_rect_range_search(node, query, range_sizes, out_data);
        }
        else {
            idx_child = node->index;
            // check if rectangular range whole in left subtree
            if (query[node->is_leaf_split_dim] + range_sizes[node->is_leaf_split_dim] <= node->split_value){
                s.push(&this->tree_nodes[idx_child]);
            }
            // check if rectangular range whole in right subtree
            else if (query[node->is_leaf_split_dim] - range_sizes[node->is_leaf_split_dim] >= node->split_value){
                s.push(&this->tree_nodes[idx_child + 1]);
            }
            else {
                // rectangular in both subtrees
                s.push(&this->tree_nodes[idx_child]);
                s.push(&this->tree_nodes[idx_child + 1]);
            }
        }
    }
}

void kd_tree::range_search_rectangular_vis(const VectorXf & query, const std::vector<float> & range_sizes, std::deque<Point *> & out_data) {
    this->vis_leaves = 0;
    std::stack<Range_search_rect_vis> s;
    int idx_child;
    s.push(Range_search_rect_vis(&this->root, this->bounding_box));

    Range_search_rect_vis str;
    std::vector<float> new_bounds;

    while (!s.empty()) {
        str = s.top();
        s.pop();

        if (str.node->is_leaf_split_dim == LEAF_NODE){
            vis_leaves ++;
            process_leaf_rect_range_search_vis(str.node, query, range_sizes, out_data, str.bounds);
        }
        else {
            idx_child = str.node->index;

            if (query[str.node->is_leaf_split_dim] + range_sizes[str.node->is_leaf_split_dim] <= str.node->split_value){
                new_bounds = str.bounds;
                new_bounds[str.node->is_leaf_split_dim * 2 + MAX] = str.node->split_value;
                s.push(Range_search_rect_vis(&this->tree_nodes[idx_child], new_bounds));
            }
            else if (query[str.node->is_leaf_split_dim] - range_sizes[str.node->is_leaf_split_dim] >= str.node->split_value){
                new_bounds = str.bounds;
                new_bounds[str.node->is_leaf_split_dim * 2 + MIN] = str.node->split_value;
                s.push(Range_search_rect_vis(&this->tree_nodes[idx_child + 1], new_bounds));
            }
            else {
                new_bounds = str.bounds;
                float old_bound = new_bounds[str.node->is_leaf_split_dim * 2 + MAX];
                new_bounds[str.node->is_leaf_split_dim * 2 + MAX] = str.node->split_value;
                s.push(Range_search_rect_vis(&this->tree_nodes[idx_child], new_bounds));
                new_bounds[str.node->is_leaf_split_dim * 2 + MAX] = old_bound;
                new_bounds[str.node->is_leaf_split_dim * 2 + MIN] = str.node->split_value;
                s.push(Range_search_rect_vis(&this->tree_nodes[idx_child + 1], new_bounds));
            }
        }
    }
}


void kd_tree::process_leaf_rect_range_search(kd_node * node, const VectorXf & query, const std::vector<float> & range_sizes, std::deque<Point *> & in_range){
    float d;
    Leaf_points lp = this->leaf_data[node->index];

    for (auto it = lp.data_begin; it != lp.data_end; it++) {
        if ((*it).p == query){
            continue;
        }

        d = dist_sqr((*it).p, query);
        (*it).query_dist = d;

        bool inside = true;
        // check whether point is in rectangular range
        for (int j = 0; j < this->dimensionality; ++j) {
            if (query[j] - range_sizes[j] <= (*it).p[j] && (*it).p[j] <= query[j] + range_sizes[j]){
            }
            else {
                inside = false;
                break;
            }
        }
        if (inside){
            in_range.push_back(&(*it));
        }
    }
}

void kd_tree::process_leaf_rect_range_search_vis(kd_node * node, const VectorXf & query, const std::vector<float> & range_sizes, std::deque<Point *> & in_range, std::vector<float> & bounds){
    float d;
    Leaf_points lp = this->leaf_data[node->index];

    this->range_search_rect_visited_leaves.push_back(bounds);

    for (auto it = lp.data_begin; it != lp.data_end; it++) {
        if ((*it).p == query){
            continue;
        }

        d = dist_sqr((*it).p, query);
        (*it).query_dist = d;

        bool inside = true;
        for (int j = 0; j < this->dimensionality; ++j) {
            if (query[j] - range_sizes[j] <= (*it).p[j] && (*it).p[j] <= query[j] + range_sizes[j]){
            }
            else {
                inside = false;
                break;
            }
        }
        if (inside){
            in_range.push_back(&(*it));
        }
    }
}

void kd_tree::range_search_spherical(const VectorXf & query, const float & rad, std::deque<Point *> & out_data) {
    this->vis_leaves = 0;
    std::vector<float> offsets = std::vector<float>(this->dimensionality);

    float rad_sqr = rad * rad;

    std::stack<Range_search_sph> s;
    s.push(Range_search_sph(&this->root, offsets, 0.0f));

    //temporal variables for search
    std::vector<float> new_off;
    Range_search_sph str;
    float old_off_split_plane;
    float new_off_split_plane;
    float new_dist;
    int idx_left;

    while (!s.empty()){
        str = s.top();
        s.pop();

        if (str.node->is_leaf_split_dim == LEAF_NODE){
            this->vis_leaves ++;
            process_leaf_sph_range_search(str.node, query, rad_sqr, out_data);
        }
        else {
            old_off_split_plane = str.off[str.node->is_leaf_split_dim];
            new_off_split_plane = query[str.node->is_leaf_split_dim] - str.node->split_value;

            idx_left = str.node->index;

            if (new_off_split_plane < 0.0f){
                //query is in left subtree

                //compute distance to farthest node
                new_dist = str.rd - old_off_split_plane * old_off_split_plane + new_off_split_plane * new_off_split_plane;

                // if farthest node still in range -> add to stack and process later
                if (new_dist < rad_sqr){
                    new_off = str.off;
                    new_off[str.node->is_leaf_split_dim] = new_off_split_plane;
                    s.push(Range_search_sph(&this->tree_nodes[idx_left + 1], new_off, new_dist));
                }
                s.push(Range_search_sph(&this->tree_nodes[idx_left], str.off, str.rd));
            }
            else {
                //query is in right subtree

                //compute distance to farthest node
                new_dist = str.rd - old_off_split_plane * old_off_split_plane + new_off_split_plane * new_off_split_plane;

                // if farthest node still in range -> add to stack and process later
                if (new_dist < rad_sqr){
                    new_off = str.off;
                    new_off[str.node->is_leaf_split_dim] = new_off_split_plane;
                    s.push(Range_search_sph(&this->tree_nodes[idx_left], new_off, new_dist));
                }
                s.push(Range_search_sph(&this->tree_nodes[idx_left + 1], str.off, str.rd));
            }
        }
    }
}

void kd_tree::range_search_spherical_vis(const VectorXf & query, const float & rad, std::deque<Point *> & out_data){
    this->vis_leaves = 0;
    std::vector<float> offsets = std::vector<float>(this->dimensionality);

    float rad_sqr = rad * rad;

    std::vector<float> b = this->bounding_box;

    std::stack<Range_search_sph_vis> s;
    s.push(Range_search_sph_vis(&this->root, offsets, 0.0f, b));

    //temporal variables for search
    std::vector<float> new_off;
    Range_search_sph_vis str;
    float old_off_split_plane;
    float new_off_split_plane;
    float new_dist;
    std::vector<float> new_bounds;
    int idx_left;

    while (!s.empty()){
        str = s.top();
        s.pop();

        if (str.node->is_leaf_split_dim == LEAF_NODE){
            this->vis_leaves ++;
            process_leaf_sph_range_search_vis(str.node, query, rad_sqr, out_data, str.bounds);
        }
        else {

            old_off_split_plane = str.off[str.node->is_leaf_split_dim];
            new_off_split_plane = query[str.node->is_leaf_split_dim] - str.node->split_value;

            idx_left = str.node->index;

            if (new_off_split_plane < 0.0f){
                //left child
                new_dist = str.rd - old_off_split_plane * old_off_split_plane + new_off_split_plane * new_off_split_plane;

                if (new_dist < rad_sqr){
                    new_off = str.off;
                    new_off[str.node->is_leaf_split_dim] = new_off_split_plane;

                    new_bounds = str.bounds;
                    new_bounds[str.node->is_leaf_split_dim * 2 + MIN] = str.node->split_value;

                    s.push(Range_search_sph_vis(&this->tree_nodes[idx_left + 1], new_off, new_dist, new_bounds));
                }

                new_bounds = str.bounds;
                new_bounds[str.node->is_leaf_split_dim * 2 + MAX] = str.node->split_value;
                s.push(Range_search_sph_vis(&this->tree_nodes[idx_left], str.off, str.rd, new_bounds));
            }
            else {
                //right child
                new_dist = str.rd - old_off_split_plane * old_off_split_plane + new_off_split_plane * new_off_split_plane;
                if (new_dist < rad_sqr){
                    new_off = str.off;
                    new_off[str.node->is_leaf_split_dim] = new_off_split_plane;

                    new_bounds = str.bounds;
                    new_bounds[str.node->is_leaf_split_dim * 2 + MAX] = str.node->split_value;

                    s.push(Range_search_sph_vis(&this->tree_nodes[idx_left], new_off, new_dist, new_bounds));
                }

                new_bounds = str.bounds;
                new_bounds[str.node->is_leaf_split_dim * 2 + MIN] = str.node->split_value;

                s.push(Range_search_sph_vis(&this->tree_nodes[idx_left + 1], str.off, str.rd, new_bounds));
            }
        }
    }
}

void kd_tree::process_leaf_sph_range_search(kd_node * node, const VectorXf & query, float & rad_sqr, std::deque<Point *> & in_range){
    float d;
    Leaf_points lp = this->leaf_data[node->index];

    for (auto it = lp.data_begin; it != lp.data_end; it++) {
        if ((*it).p == query){
            continue;
        }

        d = dist_sqr((*it).p, query);
        (*it).query_dist = d;

        // check whether point is in spherical range
        if (d < rad_sqr){
            in_range.push_back(&(*it));
        }
    }
}

void kd_tree::process_leaf_sph_range_search_vis(kd_node * node, const VectorXf & query, float & rad_sqr, std::deque<Point *> & in_range, std::vector<float> & bounds){
    float d;
    this->range_search_sph_visited_leaves.push_back(bounds);
    Leaf_points lp = this->leaf_data[node->index];

    for (auto it = lp.data_begin; it != lp.data_end; it++) {
        if ((*it).p == query){
            continue;
        }

        d = dist_sqr((*it).p, query);
        (*it).query_dist = d;

        if (d < rad_sqr){
            in_range.push_back(&(*it));
        }
    }
}

void kd_tree::destroy_tree() {
    delete [] this->tree_nodes;
    delete [] this->leaf_data;
}

// method for storing splitting planes for visualization
void kd_tree::store_vis_splitting_lines(int split_dim, float split_value, std::vector<float> & bounds) {
    if (this->dimensionality == 2){
        std::vector<float> line_point1 = std::vector<float>(this->dimensionality);
        std::vector<float> line_point2 = std::vector<float>(this->dimensionality);
        for (unsigned int i = 0; i < this->dimensionality; ++i) {
            line_point1[i] = bounds[i * 2 + MIN];
            line_point2[i] = bounds[i * 2 + MAX];
        }
        line_point1[split_dim] = split_value;
        line_point2[split_dim] = split_value;
        this->splitting_lines.push_back(line_point1);
        this->splitting_lines.push_back(line_point2);
    }
    else {
        // dimensionality is 3
        // 4 splitting_lines needed
        std::vector<float> line_point1 = std::vector<float>(this->dimensionality);
        std::vector<float> line_point2 = std::vector<float>(this->dimensionality);
        std::vector<float> line_point3 = std::vector<float>(this->dimensionality);
        std::vector<float> line_point4 = std::vector<float>(this->dimensionality);

        line_point1[split_dim] = split_value;
        line_point2[split_dim] = split_value;
        line_point3[split_dim] = split_value;
        line_point4[split_dim] = split_value;

        if (split_dim == 0){
            //up
            line_point1[1] = bounds[1 * 2 + MAX]; //back
            line_point1[2] = bounds[2 * 2 + MAX];

            line_point2[1] = bounds[1 * 2 + MIN]; //front
            line_point2[2] = bounds[2 * 2 + MAX];

            //down
            line_point3[1] = bounds[1 * 2 + MAX]; //back
            line_point3[2] = bounds[2 * 2 + MIN];

            line_point4[1] = bounds[1 * 2 + MIN]; //front
            line_point4[2] = bounds[2 * 2 + MIN];

            this->splitting_lines.push_back(line_point1);
            this->splitting_lines.push_back(line_point2);
            this->splitting_lines.push_back(line_point3);
            this->splitting_lines.push_back(line_point4);
            this->splitting_lines.push_back(line_point2);
            this->splitting_lines.push_back(line_point4);
            this->splitting_lines.push_back(line_point1);
            this->splitting_lines.push_back(line_point3);
        }
        else if (split_dim == 1){
            //up
            line_point1[0] = bounds[0 * 2 + MAX]; //right
            line_point1[2] = bounds[2 * 2 + MAX];

            line_point2[0] = bounds[0 * 2 + MIN]; //left
            line_point2[2] = bounds[2 * 2 + MAX];

            //down
            line_point3[0] = bounds[0 * 2 + MAX]; //right
            line_point3[2] = bounds[2 * 2 + MIN];

            line_point4[0] = bounds[0 * 2 + MIN]; //left
            line_point4[2] = bounds[2 * 2 + MIN];

            this->splitting_lines.push_back(line_point1);
            this->splitting_lines.push_back(line_point2);
            this->splitting_lines.push_back(line_point3);
            this->splitting_lines.push_back(line_point4);
            this->splitting_lines.push_back(line_point2);
            this->splitting_lines.push_back(line_point4);
            this->splitting_lines.push_back(line_point1);
            this->splitting_lines.push_back(line_point3);
        }
        else {
            //back
            line_point1[1] = bounds[1 * 2 + MAX]; //right
            line_point1[0] = bounds[0 * 2 + MAX];

            line_point2[1] = bounds[1 * 2 + MAX]; //left
            line_point2[0] = bounds[0 * 2 + MIN];

            //front
            line_point3[1] = bounds[1 * 2 + MIN]; //right
            line_point3[0] = bounds[0 * 2 + MAX];

            line_point4[1] = bounds[1 * 2 + MIN]; //left
            line_point4[0] = bounds[0 * 2 + MIN];

            this->splitting_lines.push_back(line_point1);
            this->splitting_lines.push_back(line_point2);
            this->splitting_lines.push_back(line_point3);
            this->splitting_lines.push_back(line_point4);
            this->splitting_lines.push_back(line_point2);
            this->splitting_lines.push_back(line_point4);
            this->splitting_lines.push_back(line_point1);
            this->splitting_lines.push_back(line_point3);
        }

    }
}

int kd_tree::get_most_extent_dimension(std::vector<float> & bounds){
    float max_extent = -1.0f;
    int most_extent_dim = 0;

    for (int i = 0; i < this->dimensionality; ++i) {
        float extent = std::abs(bounds[i * 2 + MAX] - bounds[i * 2 + MIN]);
        if (extent > max_extent){
            max_extent = extent;
            most_extent_dim = i;
        }
    }

    return most_extent_dim;
}

float kd_tree::get_splitting_value(std::vector<float> &bounds, int in_dim){
    // return middle value
    return (bounds[in_dim * 2 + MAX] - std::abs(bounds[in_dim * 2 + MAX] - bounds[in_dim * 2 + MIN])/2.0f);
}

data_iterator kd_tree::sliding_midpoint_partitioning(int split_dim, float &split_value, data_iterator data_begin,
                                                     data_iterator data_end){
    auto b = data_begin;
    auto e = data_end;
    e = std::prev(e);
    data_iterator split_it; // pivot

    float smallest_right_side = std::numeric_limits<float>::max();
    float biggest_left_side = -std::numeric_limits<float>::max();
    data_iterator sliding_it_e;
    data_iterator sliding_it_b;

    bool partition = true;

    // assume there will not be happening sliding midpoint
    bool no_sliding_midpoint = false;

    while (partition){
        while (b != data_end && (*b).p[split_dim] <= split_value){

            if (!no_sliding_midpoint){
                // iteratively find candidate for splitting plane from left direction
                if ((*b).p[split_dim] > biggest_left_side){
                    // set so far candidate with highest coords in split_dim
                    sliding_it_b = b;
                    biggest_left_side = (*b).p[split_dim];
                }
            }
            b++;
        }

        while (e != data_begin && (*e).p[split_dim] > split_value){

            if (!no_sliding_midpoint) {
                // iteratively find candidate for splitting plane from right direction
                if ((*e).p[split_dim] < smallest_right_side){
                    // set so far candidate with lowest coords in split_dim
                    sliding_it_e = e;
                    smallest_right_side = (*e).p[split_dim];
                }
            }
            e--;
        }
        if (!no_sliding_midpoint){
            if ((*e).p[split_dim] < smallest_right_side){
                sliding_it_e = e;
                smallest_right_side = (*e).p[split_dim];
            }
        }

        if (b >= e || b == data_end || e == data_begin){
            partition = false;
            split_it = b;

            if (b == data_begin){
                // all points are on right side of mid-plane -> slide to right
                split_it = sliding_it_e;
                split_value = smallest_right_side;

                std::swap((*split_it).p, (*b).p);

                return std::next(data_begin);
            }
            if (e == std::prev(data_end)){
                // all points are on left side of mid-plane -> slide to left
                split_it = sliding_it_b;
                split_value = biggest_left_side;

                std::swap((*split_it).p, (*e).p);

                return e;
            }
        }
        else {
            // we found one point which is bigger than split_value and one which is less
            // no sliding midpoint needed anymore. Just continue partitioning data
            no_sliding_midpoint = true;
            std::swap((*b).p, (*e).p);
        }
    }

    return split_it;
}