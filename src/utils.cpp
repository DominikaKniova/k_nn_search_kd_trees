#include "utils.h"

float dist_sqr(const VectorXf & p1, const VectorXf & p2){
    float sqr_norm = 0;
    for (int i = 0; i < p1.size(); ++i) {
        float diff = p1[i] - p2[i];
        sqr_norm += diff * diff;
    }
    return sqr_norm;
}