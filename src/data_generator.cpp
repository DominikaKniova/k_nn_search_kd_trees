
#include "data_generator.h"

DataGenerator::DataGenerator() {

}

void DataGenerator::generate_uniform(unsigned int size, unsigned int dimension, std::vector<float> & ranges, std::vector<Point> & out_data){
    std::random_device d;
    std::mt19937 gen(d());
    std::vector<std::uniform_real_distribution<float>> rd;

    rd.reserve(dimension);

    out_data.reserve(size);

    for (int i = 0; i < dimension; ++i) {
        rd.push_back(std::uniform_real_distribution<float> (ranges[i * 2], ranges[i * 2 + 1]));
    }

    Point point;
    point.p = std::vector<float>(dimension);
    for (int j = 0; j < size; ++j) {
        for (int i = 0; i < dimension; ++i) {
            point.p[i] = rd[i](gen);
        }
        out_data.push_back(point);
    }
}

void DataGenerator::generate_exponential(unsigned int size, unsigned int dimension, std::vector<float> & lambdas, std::vector<Point> & out_data){
    std::random_device d;
    std::mt19937 gen(d());
    std::vector<std::exponential_distribution<float>> rd;

    rd.reserve(dimension);

    out_data.reserve(size);

    for (int i = 0; i < dimension; ++i) {
        rd.push_back(std::exponential_distribution<float> (lambdas[i]));
    }

    Point point;
    point.p = std::vector<float>(dimension);
    for (int j = 0; j < size; ++j) {
        for (int i = 0; i < dimension; ++i) {
            point.p[i] = rd[i](gen);
        }
        out_data.push_back(point);
    }
}

void DataGenerator::generate_normal(unsigned int size, unsigned int dimension, std::vector<float> &means,
                                    std::vector<float> &deviations, std::vector<Point> &out_data){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<std::normal_distribution<float>> nd;
    nd.reserve(dimension);

    out_data.reserve(size);

    for (int i = 0; i < dimension; ++i) {
        nd.push_back(std::normal_distribution<float> (means[i], deviations[i]));
    }

    Point point;
    point.p = std::vector<float>(dimension);
    for (int j = 0; j < size; ++j) {
        for (int i = 0; i < dimension; ++i) {
            point.p[i] = nd[i](gen);
        }
        out_data.push_back(point);
    }
}

void DataGenerator::generate_skew_normal(unsigned int size, unsigned int dimension, std::vector<float> &locations,
                                         std::vector<float> &scales, std::vector<float> &alphas,
                                         std::vector<Point> &out_data) {
    out_data.reserve(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<boost::math::skew_normal_distribution<float>> snd;
    snd.reserve(dimension);

    std::uniform_real_distribution<float> uniform_dist(0.1f,0.99f);

    for (int i = 0; i < dimension; ++i) {
        snd.push_back(boost::math::skew_normal_distribution<float>(locations[i], scales[i], alphas[i]));
    }

    Point point;
    point.p = std::vector<float>(dimension);
    for (int j = 0; j < size; ++j) {
        for (int i = 0; i < dimension; ++i) {
            point.p[i] = boost::math::quantile(snd[i], uniform_dist(gen));
        }
        out_data.push_back(point);
    }
}

void DataGenerator::generate_circular(unsigned int size, unsigned int dimension, std::vector<float> &center,
                                      float radius, std::vector<Point> &out_data) {
    std::random_device rd;
    std::mt19937 gen(rd());

    Point point;
    point.p = std::vector<float>(dimension);
    float phi;
    float theta;

    if (dimension == 2){
        out_data.reserve(size);
        std::uniform_real_distribution<float> d(0.0f, 2.0f * M_PI);

        for (int j = 0; j < size; ++j) {
            phi = d(gen);
            point.p[0] = center[0] + radius * cos(phi);
            point.p[1] = center[1] + radius * sin(phi);
            out_data.push_back(point);
        }
    }
    else if (dimension == 3){
        out_data.reserve(size);
        std::uniform_real_distribution<float> d_phi(0.0f, 2.0f * M_PI);
        std::uniform_real_distribution<float> d_theta(0.0f, M_PI);
        for (int j = 0; j < size; ++j) {
            phi = d_phi(gen);
            theta = d_theta(gen);
            point.p[0] = center[0] + radius * cos(phi) * sin(theta);
            point.p[1] = center[1] + radius * sin(phi) * sin(theta);
            point.p[2] = center[2] + radius * cos(theta);
            out_data.push_back(point);
        }
    }
    else {
        std::vector<float> v = {0.0f, radius, 0.0f, radius,0.0f, radius, 0.0f, radius};
        generate_uniform(size, dimension, v, out_data);
    }
}
