#include "search_comparator.h"

SearchComparator::SearchComparator() {}

SearchComparator::SearchComparator(int dimensionality, kd_tree * kd_tree, std::vector<Point> & data) {
    this->dimensionality = dimensionality;
    this->data = &data;
    this->kdtree = kd_tree;
}

// script for comparing kNN search on kd tree vs naive algorithm
void SearchComparator::compare_kNN_search(const int k, const VectorXf &query, bool pq, bool vis) {
    std::deque<Point *> naive_kNNs;
    std::deque<Point *> kd_kNNs;

    if (!vis){
        // run without visualizations
        auto t_start_naive = std::chrono::high_resolution_clock::now();
        naive_k_NN(k, query, *this->data, naive_kNNs);
        auto t_end_naive = std::chrono::high_resolution_clock::now();
        std::cout << "NAIVE K_NN SEARCH: " <<std::chrono::duration<double>(t_end_naive-t_start_naive).count() << " sec" << std::endl;
        std::cout << "k-NN naive distance " << std::sqrt(dist_sqr(query, naive_kNNs[0]->p)) << std::endl;


        if (pq){
            // run with std priority queue
            std::cout << "std::priority_queue\n";
            auto t_start_kd = std::chrono::high_resolution_clock::now();
            this->kdtree->k_NN_search_priority_queue(k, query, kd_kNNs);
            auto t_end_kd = std::chrono::high_resolution_clock::now();
            std::cout << "KD K_NN SEARCH: " <<std::chrono::duration<double>(t_end_kd-t_start_kd).count() << " sec" << std::endl;
            std::cout << "kNN distance " << std::sqrt(dist_sqr(query, kd_kNNs[0]->p))<< std::endl;
            std::cout << "number of visited leaves " << this->kdtree->vis_leaves << std::endl;
        }
        else {
            // run with binomial heap
            std::cout << "binomial heap\n";
            auto t_start_kd = std::chrono::high_resolution_clock::now();
            this->kdtree->k_NN_search_binomial_heap(k, query, kd_kNNs);
            auto t_end_kd = std::chrono::high_resolution_clock::now();
            std::cout << "KD K_NN SEARCH: " <<std::chrono::duration<double>(t_end_kd-t_start_kd).count() << " sec" << std::endl;
            std::cout << "kNN kd distance " << std::sqrt(dist_sqr(query, kd_kNNs[0]->p))<< std::endl;
            std::cout << "number of visited leaves " << this->kdtree->vis_leaves << std::endl;
        }
    }
    else {
        // run with visualizations
        auto t_start_naive = std::chrono::high_resolution_clock::now();
        naive_k_NN(k, query, *this->data, naive_kNNs);
        auto t_end_naive = std::chrono::high_resolution_clock::now();
        std::cout << "NAIVE K_NN SEARCH: " <<std::chrono::duration<double>(t_end_naive-t_start_naive).count() << " sec" << std::endl;
        std::cout << "k-NN naive distance " << std::sqrt(dist_sqr(query, naive_kNNs[0]->p)) << std::endl;


        if (pq){
            std::cout << "std::priority_queue\n";
            auto t_start_kd = std::chrono::high_resolution_clock::now();
            this->kdtree->k_NN_search_priority_queue_vis(k, query, kd_kNNs);
            auto t_end_kd = std::chrono::high_resolution_clock::now();
            std::cout << "KD K_NN SEARCH: " <<std::chrono::duration<double>(t_end_kd-t_start_kd).count() << " sec" << std::endl;
            std::cout << "kNN kd distance " << std::sqrt(dist_sqr(query, kd_kNNs[0]->p))<< std::endl;
            std::cout << "number of visited leaves " << this->kdtree->vis_leaves << std::endl;
        }
        else {
            std::cout << "binomial heap\n";
            auto t_start_kd = std::chrono::high_resolution_clock::now();
            this->kdtree->k_NN_search_binomial_heap_vis(k, query, kd_kNNs);
            auto t_end_kd = std::chrono::high_resolution_clock::now();
            std::cout << "KD K_NN SEARCH: " <<std::chrono::duration<double>(t_end_kd-t_start_kd).count() << " sec" << std::endl;
            std::cout << "kNN kd distance" << std::sqrt(dist_sqr(query, kd_kNNs[0]->p))<< std::endl;
            std::cout << "number of visited leaves " << this->kdtree->vis_leaves << std::endl;
        }

        if (this->dimensionality < 4){
            int step = 1;
            if ((int)this->data->size() >= 1000){
                step = 10;
                if ((int)this->data->size() >= 100000){
                    step = ((int)this->data->size()) / 1000;
                    if ((int)this->data->size() >= 1000000) {
                        step = ((int)this->data->size()) / 100;
                    }
                }
            }
            // visualize
            std::vector<vtkSmartPointer<vtkActor>> actors;
            actors.push_back(plot_bounding_box(this->kdtree->bounding_box, this->dimensionality, &green[0]));
            actors.push_back(plot_splitting_lines(this->kdtree->splitting_lines, this->dimensionality, &red[0]));
            actors.push_back(plot_points(*this->data, this->dimensionality, step, &black[0]));
            actors.push_back(plot_visited_leaves(this->kdtree->kNN_visited_leaves, this->dimensionality));
            actors.push_back(plot_range_sph(query, std::sqrt(dist_sqr(kd_kNNs[0]->p, query)), this->dimensionality));
            actors.push_back(plot_query(query, this->dimensionality));
            actors.push_back(plot_range_points(kd_kNNs, this->dimensionality, step, &blue[0]));
            show_visualization(actors);
            actors.clear();
        }
    }

    std::cout << "\n\n";

}

// script for comparing spherical search on kd tree vs naive algorithm
void SearchComparator::compare_range_search_spherical(const float radius, const VectorXf query, bool vis) {
    std::deque<Point *> naive_range_sph;
    std::deque<Point *> kd_sph_range;

    if (vis){
        // run without visualizations
        auto t_start_naive = std::chrono::high_resolution_clock::now();
        naive_range_search_spherical(query, radius, *this->data, naive_range_sph);
        auto t_end_naive = std::chrono::high_resolution_clock::now();
        std::cout << "NAIVE RANGE SEARCH SPHERICAL: " <<std::chrono::duration<double>(t_end_naive-t_start_naive).count() << " sec" << std::endl;
        std::cout << "naive output size " << naive_range_sph.size() << std::endl;


        auto t_start_kd = std::chrono::high_resolution_clock::now();
        this->kdtree->range_search_spherical_vis(query, radius, kd_sph_range);
        auto t_end_kd = std::chrono::high_resolution_clock::now();
        std::cout << "KD RANGE SEARCH SPHERICAL: " <<std::chrono::duration<double>(t_end_kd-t_start_kd).count() << " sec" << std::endl;
        std::cout << "kd output size " << kd_sph_range.size() << std::endl;
        std::cout << "number visited leaves " << this->kdtree->vis_leaves << std::endl;

        if (this->dimensionality < 4){
            int step = 1;
            if ((int)this->data->size() >= 1000){
                step = 10;
                if ((int)this->data->size() >= 100000){
                    step = ((int)this->data->size()) / 1000;
                    if ((int)this->data->size() >= 1000000) {
                        step = ((int)this->data->size()) / 100;
                    }
                }
            }
            std::vector<vtkSmartPointer<vtkActor>> actors;
            actors.push_back(plot_bounding_box(this->kdtree->bounding_box, this->dimensionality, &green[0]));
            actors.push_back(plot_splitting_lines(this->kdtree->splitting_lines, this->dimensionality, &red[0]));
            actors.push_back(plot_points(*this->data, this->dimensionality, step, &black[0]));
            actors.push_back(plot_query(query, this->dimensionality));
            actors.push_back(plot_range_points(kd_sph_range, this->dimensionality, step, &blue[0]));
            actors.push_back(plot_range_sph(query, radius, this->dimensionality));
            actors.push_back(plot_visited_leaves(this->kdtree->range_search_sph_visited_leaves, this->dimensionality));
            show_visualization(actors);
            actors.clear();
        }

    }
    else {
        // run with visualizations
        auto t_start_naive = std::chrono::high_resolution_clock::now();
        naive_range_search_spherical(query, radius, *this->data, naive_range_sph);
        auto t_end_naive = std::chrono::high_resolution_clock::now();
        std::cout << "NAIVE RANGE SEARCH SPHERICAL: " <<std::chrono::duration<double>(t_end_naive-t_start_naive).count() << " sec" << std::endl;
        std::cout << "naive output size " << naive_range_sph.size() << std::endl;

        auto t_start_kd = std::chrono::high_resolution_clock::now();
        this->kdtree->range_search_spherical(query, radius, kd_sph_range);
        auto t_end_kd = std::chrono::high_resolution_clock::now();
        std::cout << "KD RANGE SEARCH SPHERICAL: " <<std::chrono::duration<double>(t_end_kd-t_start_kd).count() << " sec" << std::endl;
        std::cout << "kd output size " << kd_sph_range.size() << std::endl;
        std::cout << "number visited leaves " << this->kdtree->vis_leaves << std::endl;

    }

    std::cout << "\n\n";

}

// script for comparing rectangular search on kd tree vs naive algorithm
void SearchComparator::compare_range_search_rectangular(std::vector<float> &ranges, const VectorXf query, bool vis) {
    std::deque<Point *> naive_range_rect;
    std::deque<Point *> kd_rect_range;

    if (vis){
        // run with visualizations
        auto t_start_naive = std::chrono::high_resolution_clock::now();
        naive_range_search_rectangular(query, ranges, this->dimensionality, *this->data, naive_range_rect);
        auto t_end_naive = std::chrono::high_resolution_clock::now();
        std::cout << "NAIVE RANGE SEARCH RECTANGULAR: " <<std::chrono::duration<double>(t_end_naive-t_start_naive).count() << " sec" << std::endl;
        std::cout << "naive size " << naive_range_rect.size() << std::endl;

        auto t_start_kd = std::chrono::high_resolution_clock::now();
        this->kdtree->range_search_rectangular_vis(query, ranges, kd_rect_range);
        auto t_end_kd = std::chrono::high_resolution_clock::now();
        std::cout << "KD RANGE SEARCH RECTANGULAR: " <<std::chrono::duration<double>(t_end_kd-t_start_kd).count() << " sec" << std::endl;
        std::cout << "kd size " << kd_rect_range.size() << std::endl;
        std::cout << "number visited leaves " << this->kdtree->vis_leaves << std::endl;

        if (this->dimensionality < 3){
            int step = 1;
            if ((int)this->data->size() >= 1000){
                step = 10;
                if ((int)this->data->size() >= 100000){
                    step = ((int)this->data->size()) / 1000;
                    if ((int)this->data->size() >= 1000000) {
                        step = ((int)this->data->size()) / 100;
                    }
                }
            }
            std::vector<vtkSmartPointer<vtkActor>> actors;
            actors.push_back(plot_bounding_box(this->kdtree->bounding_box, this->dimensionality, &green[0]));
            actors.push_back(plot_splitting_lines(this->kdtree->splitting_lines, this->dimensionality, &red[0]));
            actors.push_back(plot_points(*this->data, this->dimensionality, step, &black[0]));
            actors.push_back(plot_query(query, this->dimensionality));
            actors.push_back(plot_range_points(kd_rect_range, this->dimensionality, step, &blue[0]));
            actors.push_back(plot_visited_leaves(this->kdtree->range_search_rect_visited_leaves, this->dimensionality));
            actors.push_back(plot_range_rect(ranges, query, this->dimensionality));
            show_visualization(actors);
            actors.clear();
        }

    }
    else {
        // run without visualizations
        auto t_start_naive = std::chrono::high_resolution_clock::now();
        naive_range_search_rectangular(query, ranges, this->dimensionality, *this->data, naive_range_rect);
        auto t_end_naive = std::chrono::high_resolution_clock::now();
        std::cout << "NAIVE RANGE SEARCH RECTANGULAR: " <<std::chrono::duration<double>(t_end_naive-t_start_naive).count() << " sec" << std::endl;
        std::cout << "naive output size " << naive_range_rect.size() << std::endl;

        auto t_start_kd = std::chrono::high_resolution_clock::now();
        this->kdtree->range_search_rectangular(query, ranges, kd_rect_range);
        auto t_end_kd = std::chrono::high_resolution_clock::now();
        std::cout << "KD RANGE SEARCH RECTANGULAR: " <<std::chrono::duration<double>(t_end_kd-t_start_kd).count() << " sec" << std::endl;
        std::cout << "kd output size " << kd_rect_range.size() << std::endl;
        std::cout << "number visited leaves " << this->kdtree->vis_leaves << std::endl;

    }

    std::cout << "\n\n";
}

// ==============================================================================
// my experiments for statistical visualizations
void print_experiment(std::string text, std::vector<double> experiment){
    std::cout << std::endl;
    std::cout << text << std::endl;
    if (experiment.size() == 7){
        // kd tree experiment
        std::cout << "kd tree \n";
        std::cout << "build time " << experiment[0] << std::endl;
        std::cout << "search time kNN " << experiment[1] << std::endl;
        std::cout << "num vis leaves kNN " << experiment[4] << std::endl;
        std::cout << "search time sph search " << experiment[2] << std::endl;
        std::cout << "num vis leaves sph search " << experiment[5] << std::endl;
        std::cout << "search time rect search " << experiment[3] << std::endl;
        std::cout << "num vis leaves rect search " << experiment[6] << std::endl;
    }
    else {
        // naive experiment
        std::cout << "naive \n";
        std::cout << "search time kNN " << experiment[0] << std::endl;
        std::cout << "search time sph search " << experiment[1] << std::endl;
        std::cout << "search time rect search " << experiment[2] << std::endl;
    }

}

std::vector<double> get_average_build_time_search_time(int dimentionality, int max_points_in_leaf, int k, std::vector<Point> & data, VectorXf query, float radius){
    int nums = 3;
    std::vector<double> times_build(nums);
    std::vector<double> times_kNN(nums);
    std::vector<double> times_sph_range(nums);
    std::vector<double> times_rect_range(nums);

    std::vector<double> num_vis_leaves_kNN(nums);
    std::vector<double> num_vis_leaves_sph(nums);
    std::vector<double> num_vis_leaves_rect(nums);

    std::deque<Point *> out_data;
    std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end;
    std::vector<float> range_sizes = {6.0f, 70.0f, 25.0f, 40.0f};

    for (int j = 0; j < nums; ++j) {
        kd_tree kdtree(dimentionality, max_points_in_leaf, false);

        // build time
        t_start = std::chrono::high_resolution_clock::now();
            kdtree.get_bounding_box(data);
            kdtree.build(data);
        t_end = std::chrono::high_resolution_clock::now();
        times_build[j] = std::chrono::duration<double>(t_end-t_start).count();

        // kNN search time
        t_start = std::chrono::high_resolution_clock::now();
            kdtree.k_NN_search_priority_queue(k, query, out_data);
        t_end = std::chrono::high_resolution_clock::now();
        times_kNN[j] = std::chrono::duration<double>(t_end-t_start).count();
        num_vis_leaves_kNN[j] = kdtree.vis_leaves;
        out_data.clear();

        // spherical range search time
        t_start = std::chrono::high_resolution_clock::now();
            kdtree.range_search_spherical(query, radius, out_data);
        t_end = std::chrono::high_resolution_clock::now();
        times_sph_range[j] = std::chrono::duration<double>(t_end-t_start).count();
        num_vis_leaves_sph[j] = kdtree.vis_leaves;
        out_data.clear();

        // rectangular range search time
        t_start = std::chrono::high_resolution_clock::now();
            kdtree.range_search_rectangular(query, range_sizes, out_data);
        t_end = std::chrono::high_resolution_clock::now();
        times_rect_range[j] = std::chrono::duration<double>(t_end-t_start).count();
        num_vis_leaves_rect[j] = kdtree.vis_leaves;
        out_data.clear();

        kdtree.destroy_tree();
    }

    std::vector<double> averages(7, 0.0);
    for (int k = 0; k < nums; ++k) {
        averages[0] += times_build[k];
        averages[1] += times_kNN[k];
        averages[2] += times_sph_range[k];
        averages[3] += times_rect_range[k];
        averages[4] += num_vis_leaves_kNN[k];
        averages[5] += num_vis_leaves_sph[k];
        averages[6] += num_vis_leaves_rect[k];
    }
    for (int k = 0; k < averages.size(); ++k) {
        averages[k] /= (double)nums;
    }

    return averages;
}

std::vector<double> get_average_search_time_naive(int dimentionality, int k, std::vector<Point> & data, VectorXf query, float radius){
    int nums = 3;
    std::vector<double> search_times_kNN(nums);
    std::vector<double> search_times_sph_range(nums);
    std::vector<double> search_times_rect_range(nums);
    std::deque<Point *> out_data;
    std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end;
    std::vector<float> range_sizes = {6.0f, 70.0f, 25.0f, 40.0f};

    for (int j = 0; j < nums; ++j) {
        // kNN search time
        t_start = std::chrono::high_resolution_clock::now();
            naive_k_NN(k, query, data, out_data);
        t_end = std::chrono::high_resolution_clock::now();
        search_times_kNN[j] = std::chrono::duration<double>(t_end-t_start).count();
        out_data.clear();

        // spherical range search time
        t_start = std::chrono::high_resolution_clock::now();
            naive_range_search_spherical(query, radius, data, out_data);
        t_end = std::chrono::high_resolution_clock::now();
        search_times_sph_range[j] = std::chrono::duration<double>(t_end-t_start).count();
        out_data.clear();

        // rectangular range search time
        t_start = std::chrono::high_resolution_clock::now();
            naive_range_search_rectangular(query, range_sizes, dimentionality, data, out_data);
        t_end = std::chrono::high_resolution_clock::now();
        search_times_rect_range[j] = std::chrono::duration<double>(t_end-t_start).count();
        out_data.clear();
    }

    std::vector<double> averages(4, 0.0);
    for (int k = 0; k < nums; ++k) {
        averages[0] += search_times_kNN[k];
        averages[1] += search_times_sph_range[k];
        averages[2] += search_times_rect_range[k];
    }
    for (int k = 0; k < averages.size(); ++k) {
        averages[k] /= (double)nums;
    }

    return averages;
}

void SearchComparator::experiment_1(unsigned int num_samples, unsigned int dimensionality, int max_points_in_leaf, int k) {
    std::cout << "\nexperiment 1\n";
    std::cout << "num samples " << num_samples << std::endl;
    std::cout << "dimensionality " << dimensionality << std::endl;
    std::cout << "max points in leaf " << max_points_in_leaf << std::endl;
    std::cout << "k " << k << std::endl;

    DataGenerator dg;
    std::vector<Point> data;
    std::vector<double> time_kd;
    std::vector<double> time_naive;
    VectorXf query;
    float radius = 50.0f;

    // uniform distribution
    std::vector<float> uniform_distr_ranges = {15.0f, 50.0f, 35.0f, 90.0f, -60.0f, 40.0f, -50.0f, 36.0f};
    dg.generate_uniform(num_samples, dimensionality, uniform_distr_ranges, data);
    query = data[(data.size() - 1) / 3].p;
    time_kd = get_average_build_time_search_time(dimensionality, max_points_in_leaf, k, data, query, radius);
    time_naive = get_average_search_time_naive(dimensionality, k, data, query, radius);
    print_experiment("uniform distribution kd", time_kd);
    print_experiment("uniform distribution naive", time_naive);
    data.clear();

    // normal distribution
    std::vector<float> mean = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> deviations = {100.0f, 50.0f, 30.0f, 70.0f};
    dg.generate_normal(num_samples, dimensionality, mean, deviations, data);
    query = data[(data.size() - 1) / 3].p;
    time_kd = get_average_build_time_search_time(dimensionality, max_points_in_leaf, k, data, query, radius);
    time_naive = get_average_search_time_naive(dimensionality, k, data, query, radius);
    print_experiment("normal distribution kd", time_kd);
    print_experiment("normal distribution naive", time_naive);
    data.clear();

    // skewed distribution
    std::vector<float> locations = {100.0f, 250.0f, 150.0f, 300.0f};
    std::vector<float> scales = {0.5, 0.75f, 0.25f, 0.5f};
    std::vector<float> alphas = {20.0f, 40.0f, 10.0f, 40.0f};
    dg.generate_skew_normal(num_samples, dimensionality, locations, scales, alphas, data);
    query = data[(data.size() - 1) / 3].p;
    time_kd = get_average_build_time_search_time(dimensionality, max_points_in_leaf, k, data, query, radius);
    time_naive = get_average_search_time_naive(dimensionality, k, data, query, radius);
    print_experiment("skewed normal distribution kd", time_kd);
    print_experiment("skewed normal distribution naive", time_naive);
    data.clear();

    // circle/spherical distribution (only if dimensionality < 4)
    if (dimensionality < 4){
        std::vector<float> center = {1.0f, 2.0f, 3.0f, 4.0f};
        dg.generate_circular(num_samples, dimensionality, query, radius * 0.75f, data);
        query = center;
        time_kd = get_average_build_time_search_time(dimensionality, max_points_in_leaf, k, data, query, radius);
        time_naive = get_average_search_time_naive(dimensionality, k, data, query, radius);
        print_experiment("circle/sphere distribution kd", time_kd);
        print_experiment("circle/sphere distribution naive", time_naive);
        data.clear();
    }

    std::vector<float> lambda = {0.5f, 0.75f, 1.0f, 0.25f};
    dg.generate_exponential(num_samples, dimensionality, lambda, data);
    query = data[(data.size() - 1) / 3].p;
    time_kd = get_average_build_time_search_time(dimensionality, max_points_in_leaf, k, data, query, radius);
    time_naive = get_average_search_time_naive(dimensionality, k, data, query, radius);
    print_experiment("exponential distribution kd", time_kd);
    print_experiment("exponential distribution naive", time_naive);
    data.clear();
}

unsigned int samples_sizes [8] = {1000, 5000, 10000, 50000, 100000, 400000, 800000, 1000000};
unsigned int max_leaf_points_sizes [9] = {10, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000};
int num_mlp = 9;int num_dim = 3;
unsigned int dimensions [3] = {2, 3, 4};
int num_sample_sizes = 8;
std::string path_exp1 = "./graph_data/exp1/";
std::string path_exp2 = "./graph_data/exp2/";
std::string path_exp4 = "./graph_data/exp4/";



double get_average_build_time(unsigned int dim , unsigned int max_leaf_points, std::vector<Point> & data){
    int nums = 3;
    std::vector<double> times_build(nums);

    std::deque<Point *> out_data;
    std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end;

    for (int j = 0; j < nums; ++j) {
        kd_tree kdtree(dim, max_leaf_points, false);
        // build time
        t_start = std::chrono::high_resolution_clock::now();
            kdtree.get_bounding_box(data);
            kdtree.build(data);
        t_end = std::chrono::high_resolution_clock::now();
        times_build[j] = std::chrono::duration<double>(t_end-t_start).count();

        kdtree.destroy_tree();
    }

    double average = 0.0f;
    for (int k = 0; k < nums; ++k) {
        average += times_build[k];
    }
    average /= (float)nums;

    return average;
}


void experiment_1_uniform_fix_dim_mlp_var_N_dparams(unsigned int dim, unsigned int max_leaf_point){
    std::ofstream out_file;
    std::string f_name;
    f_name += path_exp1;
    f_name += "uniform_" + std::to_string(dim) + "_" + std::to_string(max_leaf_point) + ".txt";
    out_file.open(f_name, std::ios::out);


    std::vector<float> uniform_distr_ranges1 = {15.0f, 50.0f, 35.0f, 90.0f, -60.0f, 40.0f, -50.0f, 36.0f};
    std::vector<float> uniform_distr_ranges2 = {-200.0f, 200.0f, -200.0f, 200.0f, -200.0f, 200.0f, -200.0f, 200.0f,};
    std::vector<float> uniform_distr_ranges3 = {-200.0f, 200.0f, 0.0f, 10.0f, -200.0f, 200.0f, 50.0f, 55.0f};
    std::vector<std::vector<float>> ranges = {uniform_distr_ranges1, uniform_distr_ranges2, uniform_distr_ranges3};

    DataGenerator dg;
    std::vector<Point> data;
    double time;
    std::vector<double> time_naive;

    out_file << ranges.size() << " " << dim << " " << num_sample_sizes << " " << max_leaf_point <<"\n";

    for (int j = 0; j < ranges.size(); ++j) {
        for (int i = 0; i < num_sample_sizes; ++i) {
            dg.generate_uniform(samples_sizes[i], dim, ranges[j], data);
            time = get_average_build_time(dim, max_leaf_point, data);
            out_file << samples_sizes[i] << " " <<  time << "\n";
            data.clear();
        }
    }

    out_file.close();
}

void experiment_1_normal_fix_dim_mlp_var_N_dparams(unsigned int dim, unsigned int max_leaf_point){
    std::ofstream out_file;
    std::string f_name;
    f_name += path_exp1;
    f_name += "normal_" + std::to_string(dim) + "_" + std::to_string(max_leaf_point) + ".txt";
    out_file.open(f_name, std::ios::out);

    std::vector<float> mean1 = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> deviations1 = {100.0f, 100.0f, 100.0f, 100.0f};
    std::vector<float> mean2 = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> deviations2 = {100.0f, 5.0f, 100.0f, 10.0f};
    std::vector<float> mean3 = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> deviations3 = {200.0f, 50.0f, 30.0f, 20.0f};

    std::vector<std::vector<float>> params = {mean1, mean2, mean3, deviations1, deviations2, deviations3};

    DataGenerator dg;
    std::vector<Point> data;
    double time;
    std::vector<double> time_naive;
    int num_params = params.size()/2;

    out_file << num_params << " " << dim << " " << num_sample_sizes << " " << max_leaf_point <<"\n";

    for (int j = 0; j < num_params; ++j) {
        for (int i = 0; i < num_sample_sizes; ++i) {
            dg.generate_normal(samples_sizes[i], dim, params[j], params[j + num_params], data);
            time = get_average_build_time(dim, max_leaf_point, data);
            out_file << samples_sizes[i] << " " <<  time << "\n";
            data.clear();
        }
    }

    out_file.close();
}

void experiment_1_skewnormal_fix_dim_mlp_var_N_dparams(unsigned int dim, unsigned int max_leaf_point){
    std::ofstream out_file;
    std::string f_name;
    f_name += path_exp1;
    f_name += "skewnormal_" + std::to_string(dim) + "_" + std::to_string(max_leaf_point) + ".txt";
    out_file.open(f_name, std::ios::out);

    std::vector<float> locations = {100.0f, 250.0f, 150.0f, 300.0f};
    std::vector<float> scales = {0.5, 0.75f, 0.25f, 0.5f};
    std::vector<float> alphas = {20.0f, 40.0f, 10.0f, 40.0f};
    std::vector<float> locations2 = {10.0f, 10.0f, 10.0f, 10.0f};
    std::vector<float> scales2 = {0.25, 0.25f, 0.25f, 0.25f};
    std::vector<float> alphas2 = {-20.0f, -40.0f, -10.0f, 40.0f};

    DataGenerator dg;
    std::vector<Point> data;
    double time;
    std::vector<double> time_naive;
    int num_params = 2;

    out_file << num_params << " " << dim << " " << num_sample_sizes << " " << max_leaf_point <<"\n";

    std::vector<std::vector<float>> params = {locations, scales, alphas, locations2, scales2, alphas2};

    for (int j = 0; j < num_params; ++j) {
    	std::cout << "params " << j << std::endl;
        for (int i = 0; i < num_sample_sizes; ++i) {
            std::cout << samples_sizes[i] << std::endl;
            dg.generate_skew_normal(samples_sizes[i], dim, params[j * 3], params[j * 3 + 1], params[j * 3 + 2], data);
            time = get_average_build_time(dim, max_leaf_point, data);
            out_file << samples_sizes[i] << " " <<  time << "\n";
            data.clear();
        }
    }

    out_file.close();
}

void experiment_1_exponential_fix_dim_mlp_var_N_dparams(unsigned int dim, unsigned int max_leaf_point){
    std::ofstream out_file;
    std::string f_name;
    f_name += path_exp1;
    f_name += "exponential_" + std::to_string(dim) + "_" + std::to_string(max_leaf_point) + ".txt";
    out_file.open(f_name, std::ios::out);

    std::vector<float> lambda1 = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> lambda2 = {10.0f, 0.5f, 0.5f, 10.0f};
    std::vector<float> lambda3 = {100.0f, 0.5f, 0.5f, 0.5f};
    std::vector<std::vector<float>> params = {lambda1, lambda2, lambda3};

    DataGenerator dg;
    std::vector<Point> data;
    double time;
    std::vector<double> time_naive;
    int num_params = 3;

    out_file << num_params << " " << dim << " " << num_sample_sizes << " " << max_leaf_point <<"\n";

    for (int j = 0; j < num_params; ++j) {
        for (int i = 0; i < num_sample_sizes; ++i) {
            std::cout << samples_sizes[i] << std::endl;
            dg.generate_exponential(samples_sizes[i], dim, params[j], data);
            time = get_average_build_time(dim, max_leaf_point, data);
            out_file << samples_sizes[i] << " " <<  time << "\n";
            data.clear();
        }
    }

    out_file.close();
}



void SearchComparator::experiment_1_build_diff_distributions_fix_dim_mlp_var_N_dparams(unsigned int dim, unsigned  int max_leaf_point) {
  std::cout << "uniform experiment\n";
  experiment_1_uniform_fix_dim_mlp_var_N_dparams(dim, max_leaf_point);
  std::cout << "normal experiment\n";
  experiment_1_normal_fix_dim_mlp_var_N_dparams(dim, max_leaf_point);
   std::cout << "skew normal experiment\n";
   experiment_1_skewnormal_fix_dim_mlp_var_N_dparams(dim, max_leaf_point);
  std::cout << "exponential experiment\n";
  experiment_1_exponential_fix_dim_mlp_var_N_dparams(dim, max_leaf_point);
}

void experiment_2_uniform_fix_Nfname_dim_var_mlp(unsigned int num_samples, unsigned int dim){
    std::ofstream out_file;
    std::string f_name;
    f_name += path_exp2;
    f_name += "uniform_" + std::to_string(dim) + "_" + std::to_string(num_samples) + ".txt";
    out_file.open(f_name, std::ios::out);

    std::vector<float> uniform_distr_ranges = {15.0f, 50.0f, 35.0f, 90.0f, -60.0f, 40.0f, -50.0f, 36.0f};

    DataGenerator dg;
    std::vector<Point> data;
    double time;

    out_file << dim << " " << num_mlp << " " << num_samples <<"\n";

    for (int i = 0; i < num_mlp; ++i) {
        dg.generate_uniform(num_samples, dim, uniform_distr_ranges, data);
        time = get_average_build_time(dim, max_leaf_points_sizes[i], data);
        out_file << max_leaf_points_sizes[i] << " " <<  time << "\n";
        data.clear();
    }
    out_file.close();
}

void experiment_2_normal_fix_N_dim_var_mlp(unsigned int num_samples, unsigned int dim){
    std::ofstream out_file;
    std::string f_name;
    f_name += path_exp2;
    f_name += "normal_" + std::to_string(dim) + "_" + std::to_string(num_samples) + ".txt";
    out_file.open(f_name, std::ios::out);

    std::vector<float> mean = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> deviation = {100.0f, 30.0f, 80.0f, 50.0f};

    DataGenerator dg;
    std::vector<Point> data;
    double time;

    out_file << dim << " " << num_mlp << " " << num_samples <<"\n";

    for (int i = 0; i < num_mlp; ++i) {
        dg.generate_normal(num_samples, dim, mean, deviation, data);
        time = get_average_build_time(dim, max_leaf_points_sizes[i], data);
        out_file << max_leaf_points_sizes[i] << " " <<  time << "\n";
        data.clear();
    }
    out_file.close();
}

void experiment_2_skewnormal_fix_N_dim_var_mlp(unsigned int num_samples, unsigned int dim){
    std::ofstream out_file;
    std::string f_name;
    f_name += path_exp2;
    f_name += "skewNormal_" + std::to_string(dim) + "_" + std::to_string(num_samples) + ".txt";
    out_file.open(f_name, std::ios::out);

    std::vector<float> locations = {100.0f, 250.0f, 150.0f, 300.0f};
    std::vector<float> scales = {0.5, 0.75f, 0.25f, 0.5f};
    std::vector<float> alphas = {20.0f, -40.0f, -10.0f, 10.0f};

    DataGenerator dg;
    std::vector<Point> data;
    double time;

    out_file << dim << " " << num_mlp << " " << num_samples <<"\n";

    for (int i = 0; i < num_mlp; ++i) {
        dg.generate_skew_normal(num_samples, dim, locations, scales, alphas, data);
        time = get_average_build_time(dim, max_leaf_points_sizes[i], data);
        out_file << max_leaf_points_sizes[i] << " " <<  time << "\n";
        data.clear();
    }
    out_file.close();
}

void experiment_2_exponential_fix_N_dim_var_mlp(unsigned int num_samples, unsigned int dim){
    std::ofstream out_file;
    std::string f_name;
    f_name += path_exp2;
    f_name += "exponential_" + std::to_string(dim) + "_" + std::to_string(num_samples) + ".txt";
    out_file.open(f_name, std::ios::out);

    std::vector<float> lambda = {50.0f, 10.5f, 30.5f, 20.0f};
    
    DataGenerator dg;
    std::vector<Point> data;
    double time;

    out_file << dim << " " << num_mlp << " " << num_samples <<"\n";

    for (int i = 0; i < num_mlp; ++i) {
		dg.generate_exponential(num_samples, dim, lambda, data);
        time = get_average_build_time(dim, max_leaf_points_sizes[i], data);
        out_file << max_leaf_points_sizes[i] << " " <<  time << "\n";
        data.clear();
    }
    out_file.close();
}

void SearchComparator::experiment_2_build_diff_distr_fix_N_dim_var_mlp(unsigned int num_samples, unsigned int dim) {
    // std::cout << "uniform experiment\n";
    // experiment_2_uniform_fix_Nfname_dim_var_mlp(num_samples, dim);
    // std::cout << "normal experiment\n";
    // experiment_2_normal_fix_N_dim_var_mlp(num_samples, dim);
    // std::cout << "skew normal experiment\n";
    // experiment_2_skewnormal_fix_N_dim_var_mlp(num_samples, dim);
    std::cout << "exponential experiment\n";
    experiment_2_exponential_fix_N_dim_var_mlp(num_samples, dim);
}

void experiment4_average_search_times_kd(int dimenstionality, int max_points_in_leaf, VectorXf query,
                                         std::vector<int> &ks, std::vector<float> &radii,
                                         std::vector<std::vector<float>> &rect_range, std::vector<Point> &data,
                                         std::string dist, bool binomial_heap){

    int nums = 3;
    std::vector<double> times_kNN(nums);
    std::vector<double> times_sph_range(nums);
    std::vector<double> times_rect_range(nums);
    std::vector<int> num_vis_leaves_kNN(nums);
    std::vector<int> num_vis_leaves_sph(nums);
    std::vector<int> num_vis_leaves_rect(nums);
    std::deque<Point *> out_data;
    std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end;

    kd_tree kdtree(dimenstionality, max_points_in_leaf, false);
    kdtree.get_bounding_box(data);
    kdtree.build(data);

    std::vector<double> average_times(ks.size());
    // kNN search time
    for (int l = 0; l < ks.size(); ++l) {
        for (int j = 0; j < nums; ++j) {
            t_start = std::chrono::high_resolution_clock::now();
            if (!binomial_heap){
                kdtree.k_NN_search_priority_queue(ks[l], query, out_data);
            }
            else {
                kdtree.k_NN_search_binomial_heap(ks[l], query, out_data);
            }
            t_end = std::chrono::high_resolution_clock::now();
            times_kNN[j] = std::chrono::duration<double>(t_end-t_start).count();
            num_vis_leaves_kNN[l] = kdtree.vis_leaves;
            out_data.clear();
        }

        double average_time = 0.0f;
        for (int k = 0; k < nums; ++k) {
            average_time += times_kNN[k];
        }
        average_time /= (double) nums;

        average_times[l] = average_time;
    }
    // print to file
    std::ofstream out_file;
    std::string f_name;
    f_name += path_exp4;
    if (!binomial_heap){
        f_name += "kNN_"+ dist + "_" + std::to_string(dimenstionality) + "_mpl" + std::to_string(max_points_in_leaf) + "_pq" + ".txt";
    }
    else {
        f_name += "kNN_"+ dist + "_" + std::to_string(dimenstionality) + "_mpl" + std::to_string(max_points_in_leaf) + "_bh" + ".txt";
    }
    out_file.open(f_name, std::ios::out);

    out_file << dimenstionality << " " << data.size() << " " << max_points_in_leaf << " " << ks.size() << "\n";

    for (int m = 0; m < ks.size(); ++m) {
        out_file << ks[m] << " " << average_times[m] << " " << num_vis_leaves_kNN[m] << "\n";
    }

    out_file.close();
    //==================================================================================================================

    average_times = std::vector<double>(radii.size()) ;
    // spherical search time
    for (int l = 0; l < radii.size(); ++l) {
        for (int j = 0; j < nums; ++j) {
            t_start = std::chrono::high_resolution_clock::now();
                kdtree.range_search_spherical(query, radii[l], out_data);
            t_end = std::chrono::high_resolution_clock::now();
            times_sph_range[j] = std::chrono::duration<double>(t_end-t_start).count();
            num_vis_leaves_sph[l] = kdtree.vis_leaves;
            out_data.clear();
        }

        double average_time = 0.0f;
        for (int k = 0; k < nums; ++k) {
            average_time += times_sph_range[k];
        }
        average_time /= (double) nums;

        average_times[l] = average_time;
    }
    // print to file
    std::ofstream out_file1;
    std::string f_name1;
    f_name1 += path_exp4;
    f_name1 += "sph_"+ dist + "_" + std::to_string(dimenstionality) + "_mpl" + std::to_string(max_points_in_leaf) + ".txt";
    out_file1.open(f_name1, std::ios::out);
    out_file1 << dimenstionality << " " << data.size() << " " << max_points_in_leaf << " " << radii.size() << "\n";
    for (int m = 0; m < radii.size(); ++m) {
        out_file1 << radii[m] << " " << average_times[m] << " " << num_vis_leaves_sph[m] << "\n";
    }
    out_file1.close();
    //==================================================================================================================

    average_times = std::vector<double>(rect_range.size()) ;
    // rectangular search time
    for (int l = 0; l < rect_range.size(); ++l) {
        for (int j = 0; j < nums; ++j) {
            t_start = std::chrono::high_resolution_clock::now();
                kdtree.range_search_rectangular(query, rect_range[l], out_data);
            t_end = std::chrono::high_resolution_clock::now();
            times_rect_range[j] = std::chrono::duration<double>(t_end-t_start).count();
            num_vis_leaves_rect[l] = kdtree.vis_leaves;
            out_data.clear();
        }

        double average_time = 0.0f;
        for (int k = 0; k < nums; ++k) {
            average_time += times_rect_range[k];
        }
        average_time /= (double) nums;

        average_times[l] = average_time;
    }
    // print to file
    std::ofstream out_file2;
    std::string f_name2;
    f_name2 += path_exp4;
    f_name2 += "rect_"+ dist + "_" + std::to_string(dimenstionality) + "_mpl" + std::to_string(max_points_in_leaf) + ".txt";
    out_file2.open(f_name2, std::ios::out);
    out_file2 << dimenstionality << " " << data.size() << " " << max_points_in_leaf << " " << rect_range.size() << "\n";
    for (int m = 0; m < rect_range.size(); ++m) {
        out_file2 << average_times[m] << " " << num_vis_leaves_rect[m] << "\n";
    }
    out_file2.close();
    //==================================================================================================================

    kdtree.destroy_tree();
}

void experiment4_average_search_times_naive(int dimensionality, VectorXf query,
                                         std::vector<int> &ks, std::vector<float> &radii,
                                         std::vector<std::vector<float>> &rect_range, std::vector<Point> &data,
                                         std::string dist){

    int nums = 3;
    std::vector<double> times_kNN(nums);
    std::vector<double> times_sph_range(nums);
    std::vector<double> times_rect_range(nums);
    std::deque<Point *> out_data;
    std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end;

    std::vector<double> average_times(ks.size());
    // kNN search time
    for (int l = 0; l < ks.size(); ++l) {
        for (int j = 0; j < nums; ++j) {
            t_start = std::chrono::high_resolution_clock::now();
                naive_k_NN(ks[l], query, data, out_data);
            t_end = std::chrono::high_resolution_clock::now();
            times_kNN[j] = std::chrono::duration<double>(t_end-t_start).count();
            out_data.clear();
        }

        double average_time = 0.0f;
        for (int k = 0; k < nums; ++k) {
            average_time += times_kNN[k];
        }
        average_time /= (double) nums;

        average_times[l] = average_time;
    }
    // print to file
    std::ofstream out_file;
    std::string f_name;
    f_name += path_exp4;
        f_name += "naive_kNN_"+ dist + "_" + std::to_string(dimensionality) + ".txt";
    out_file.open(f_name, std::ios::out);

    out_file << dimensionality << " " << data.size() << " " << ks.size() << "\n";

    for (int m = 0; m < ks.size(); ++m) {
        out_file << ks[m] << " " << average_times[m] << "\n";
    }

    out_file.close();
    //==================================================================================================================

    average_times = std::vector<double>(radii.size()) ;
    // spherical search time
    for (int l = 0; l < radii.size(); ++l) {
        for (int j = 0; j < nums; ++j) {
            t_start = std::chrono::high_resolution_clock::now();
                naive_range_search_spherical(query, radii[l], data, out_data);
            t_end = std::chrono::high_resolution_clock::now();
            times_sph_range[j] = std::chrono::duration<double>(t_end-t_start).count();
            out_data.clear();
        }

        double average_time = 0.0f;
        for (int k = 0; k < nums; ++k) {
            average_time += times_sph_range[k];
        }
        average_time /= (double) nums;

        average_times[l] = average_time;
    }
    // print to file
    std::ofstream out_file1;
    std::string f_name1;
    f_name1 += path_exp4;
    f_name1 += "naive_sph_"+ dist + "_" + std::to_string(dimensionality) + ".txt";
    out_file1.open(f_name1, std::ios::out);
    out_file1 << dimensionality << " " << data.size()  << " " << radii.size() << "\n";
    for (int m = 0; m < radii.size(); ++m) {
        out_file1 << radii[m] << " " << average_times[m] << "\n";
    }
    out_file1.close();
    //==================================================================================================================

    average_times = std::vector<double>(rect_range.size()) ;
    // rectangular search time
    for (int l = 0; l < rect_range.size(); ++l) {
        for (int j = 0; j < nums; ++j) {
            t_start = std::chrono::high_resolution_clock::now();
                naive_range_search_rectangular(query, rect_range[l], dimensionality, data, out_data);
            t_end = std::chrono::high_resolution_clock::now();
            times_rect_range[j] = std::chrono::duration<double>(t_end-t_start).count();
            out_data.clear();
        }

        double average_time = 0.0f;
        for (int k = 0; k < nums; ++k) {
            average_time += times_rect_range[k];
        }
        average_time /= (double) nums;

        average_times[l] = average_time;
    }
    // print to file
    std::ofstream out_file2;
    std::string f_name2;
    f_name2 += path_exp4;
    f_name2 += "naive_rect_"+ dist + "_" + std::to_string(dimensionality) + ".txt";
    out_file2.open(f_name2, std::ios::out);
    out_file2 << dimensionality << " " << data.size() << " " << rect_range.size() << "\n";
    for (int m = 0; m < rect_range.size(); ++m) {
        out_file2  << average_times[m] << "\n";
    }
    out_file2.close();
    //==================================================================================================================

}

void SearchComparator::experiment_4_kNN_fix_N_dim_k_var_mlp(unsigned int num_samples, unsigned int dim,
                                                            std::vector<int> & ks, std::vector<float> & radii, std::vector<std::vector<float>> & rect_range, bool binom_heap) {
    DataGenerator dg = DataGenerator();
    std::vector<Point> data;

    // uniform distr
    std::cout << "uniform\n";
    std::vector<float> uniform_distr_ranges = {15.0f, 50.0f, 35.0f, 90.0f, -60.0f, 40.0f, -50.0f, 36.0f};
    dg.generate_uniform(num_samples, dim, uniform_distr_ranges, data);
    VectorXf query = data[data.size() / 3].p;
    for (int j = 0; j < num_mlp; ++j) {
        std::cout <<max_leaf_points_sizes[j] << std::endl;
        experiment4_average_search_times_kd(dim, max_leaf_points_sizes[j], query, ks, radii,
                                            rect_range, data, "uniform", binom_heap);
    }
    experiment4_average_search_times_naive(dim, query, ks, radii, rect_range, data, "uniform");
    data.clear();

    // normal
    std::cout << "normal\n";
    std::vector<float> mean = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> deviation = {100.0f, 30.0f, 80.0f, 50.0f};
    dg.generate_normal(num_samples, dim, mean, deviation, data);
    query = data[data.size() / 3].p;
    for (int j = 0; j < num_mlp; ++j) {
        std::cout <<max_leaf_points_sizes[j] << std::endl;
        experiment4_average_search_times_kd(dim, max_leaf_points_sizes[j], query, ks, radii,
                                            rect_range, data, "normal", binom_heap);
    }
    experiment4_average_search_times_naive(dim, query, ks, radii, rect_range, data, "normal");
    data.clear();

    // skew normal
    std::cout << "skew normal\n";
    std::vector<float> locations = {100.0f, 250.0f, 150.0f, 300.0f};
    std::vector<float> scales = {0.5, 0.75f, 0.25f, 0.5f};
    std::vector<float> alphas = {20.0f, -40.0f, -10.0f, 10.0f};
    dg.generate_skew_normal(num_samples, dim, locations, scales, alphas, data);
    query = data[data.size() / 3].p;
    for (int j = 0; j < num_mlp; ++j) {
        std::cout <<max_leaf_points_sizes[j] << std::endl;
        experiment4_average_search_times_kd(dim, max_leaf_points_sizes[j], query, ks, radii,
                                            rect_range, data, "skewNormal", binom_heap);
    }
    experiment4_average_search_times_naive(dim, query, ks, radii, rect_range, data, "skewNormal");
    data.clear();

    //exponential
    std::cout << "exp\n";
    std::vector<float> lambda = {50.0f, 10.5f, 30.5f, 20.0f};
    dg.generate_exponential(num_samples, dim, lambda, data);
    query = data[data.size() / 3].p;
    for (int j = 0; j < num_mlp; ++j) {
        std::cout <<max_leaf_points_sizes[j] << std::endl;
        experiment4_average_search_times_kd(dim, max_leaf_points_sizes[j], query, ks, radii,
                                            rect_range, data, "exponential", binom_heap);
    }
    experiment4_average_search_times_naive(dim, query, ks, radii, rect_range, data, "exponential");
    data.clear();
}
