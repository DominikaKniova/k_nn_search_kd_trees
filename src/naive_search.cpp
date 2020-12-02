#include <limits>
#include <cmath>
#include "naive_search.h"

// naive implementation of kNN search
void naive_k_NN(const int & k, const VectorXf & query, std::vector<Point> & data, std::deque<Point *> & out_data){
    float kNN_dist = std::numeric_limits<float>::max();

    //farthest point is first in the priority queue
    std::priority_queue<Point *, std::vector<Point *>, kd_search_comparator> kNN_q;

    float d;
    for (unsigned int i = 0; i < data.size(); ++i) {
        if (data[i].p != query){
            d = dist_sqr(data[i].p, query);
            data[i].query_dist = d;

            if (kNN_q.size() < k){
                kNN_q.push(&data[i]);
            }
            else if (d < kNN_dist){
                kNN_q.push(&data[i]);
                if (kNN_q.size() > k){
                    kNN_q.pop();
                    kNN_dist = kNN_q.top()->query_dist;
                }
            }
        }
    }

    for (unsigned int i = 0; i < k; ++i) {
        out_data.push_back(kNN_q.top());
        kNN_q.pop();
    }
}

// naive implementation of spherical search
void naive_range_search_spherical(const VectorXf &query, const float rad, std::vector<Point> & data, std::deque<Point *> & out_data){
    // it is more effective to only compute with squared radius
    float rad_sqr = rad * rad;

    float d;
    for (unsigned int i = 0; i < data.size(); ++i) {
        if (data[i].p != query){
            d = dist_sqr(data[i].p, query);
            data[i].query_dist = d;

            if (d < rad_sqr){
                out_data.push_back(&data[i]);
            }
        }
    }
}

// naive implementation of rectangular search
void naive_range_search_rectangular(const VectorXf &query, const std::vector<float> &range_sizes, const int dimensionality, std::vector<Point> & data, std::deque<Point *> & out_data){
    float d;
    for (unsigned int i = 0; i < data.size(); ++i) {
        if (data[i].p == query){
            continue;
        }

        d = dist_sqr(data[i].p, query);
        data[i].query_dist = d;

        bool inside = true;
        for (int j = 0; j < dimensionality; ++j) {
            if (query[j] - range_sizes[j] <= data[i].p[j] && data[i].p[j] <= query[j] + range_sizes[j]){
            }
            else {
                inside = false;
                break;
            }
        }
        if (inside){
            out_data.push_back(&data[i]);
        }
    }
}