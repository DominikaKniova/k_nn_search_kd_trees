
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <istream>
#include <sstream>
#include "point.h"
#include "data_generator.h"
#include "kd_node.h"
#include "naive_search.h"
#include "search_comparator.h"
#include "visualizer.h"

std::vector<float> parse_distr_params(int min, int max, std::vector<std::string> & data){
	std::vector<float> result;
	if (data.size() - 1 < min || data.size() - 1 > max)
	{
		return result;
	}
	for(int i = 1; i < data.size(); i++) {
			result.push_back(std::stof(data[i]));
	}
	return result;
}

std::vector<std::string> parse_search_params(int min, int max, std::vector<std::string> & data){
	std::vector<std::string> result;
	if (data.size() - 1 < min || data.size() - 1 > max)
	{
		return result;
	}
	for(int i = 1; i < data.size(); i++)
	{
		result.push_back(data[i]);
	}
	return result;
}

int main() {
	// processing inputs (parsing) and running kd tree algorithms
    DataGenerator dg = DataGenerator();

    std::string line, data_info, distr_info, num_searches;
    std::vector<std::string> line_content;
    std::string line_element;


    if (!std::getline(std::cin, data_info)){
    	std::cout << "wrong input\n";
    	return 0;
    }

    if (!std::getline(std::cin, distr_info) ){
		std::cout << "wrong input\n";
		return 0;
	}
	if (!std::getline(std::cin, num_searches)){
		std::cout << "wrong input\n";
		return 0;
	}

    std::vector<int> data_params;
    std::stringstream ss(data_info);
    while (std::getline(ss, line_element, ' ')){
    	data_params.push_back(std::stoi(line_element));
    }
    if (data_params.size() != 4){
    	std::cout << "wrong input\n";
    	return 0;
    }

    int num_samples = data_params[0];
    int dim = data_params[1];
    int mlp = data_params[2];
    bool visualize = data_params[3];
    if (dim > 3){
    	visualize = false;
    }

    std::vector<std::string> dist_params;
	std::stringstream ss2(distr_info);
    while (std::getline(ss2, line_element, ' ')){
    	dist_params.push_back(line_element);
    }
    if (dist_params.empty()){
    	std::cout << "wrong input\n";
    	return 0;
    }

    std::vector<Point> data;
    std::vector<float> distribution_params;
	std::cout << "generating samples" << std::endl;


	// base on input parameter decide which distribution to choose and generate data with distribution parameters
	if (dist_params[0] == "uniform"){
		distribution_params = parse_distr_params(4, 8, dist_params);
		if (!distribution_params.size()){
			std::cout << "wrong input\n";
			return 0;
		}
		dg.generate_uniform(num_samples, dim, distribution_params, data);
	}
	else if (dist_params[0] == "normal"){
		distribution_params = parse_distr_params(4, 8, dist_params);
		if (!distribution_params.size()){
			std::cout << "wrong input\n";
			return 0;
		}
		std::vector<float> mean;
		std::vector<float> deviation;
		for (int i = 0; i < dim; ++i)
		{
			mean.push_back(distribution_params[i]);
			deviation.push_back(distribution_params[i + dim]);
		}
        dg.generate_normal(num_samples, dim, mean, deviation, data);
	}
	else if (dist_params[0] == "skewnormal"){
		distribution_params = parse_distr_params(6, 12, dist_params);
		if (!distribution_params.size()){
			std::cout << "wrong input\n";
			return 0;
		}
		std::vector<float> locations;
		std::vector<float> scales;
		std::vector<float> alphas;
		for (int i = 0; i < dim; ++i)
		{
			locations.push_back(distribution_params[i]);
			scales.push_back(distribution_params[i + dim]);
			alphas.push_back(distribution_params[i + 2 * dim]);
		}
        dg.generate_skew_normal(num_samples, dim, locations, scales, alphas, data);

	}
	else if (dist_params[0] == "exponential"){
		distribution_params = parse_distr_params(2, 4, dist_params);
		if (!distribution_params.size()){
			std::cout << "wrong input\n";
			return 0;
		}
		dg.generate_exponential(num_samples, dim, distribution_params, data);
	}
	else if (dist_params[0] == "circle"){
		distribution_params = parse_distr_params(3, 5, dist_params);
		if (!distribution_params.size()){
			std::cout << "wrong input\n";
			return 0;
		}
		std::vector<float> center;
		for (int i = 0; i < dim; ++i)
		{
			center.push_back(distribution_params[i]);
		}
		float radius = std::stof(dist_params[dim + 1]);
        dg.generate_circular(num_samples, dim, center, radius, data);
	}
	else {
		std::cout << "wrong input\n";
		return 0;
	}

    int num_search  = std::stoi(num_searches);
    if (num_search > 3) {
    	std::cout << "wrong input\n";
    	return 0;
    }

    // build tree with generated data
    // print build time
    kd_tree kdtree = kd_tree(dim, mlp, visualize);
    auto t_start= std::chrono::high_resolution_clock::now();
    kdtree.build(data);
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "BUILD TIME: " <<std::chrono::duration<double>(t_end-t_start).count() << " sec" << std::endl;
    std::cout << std::endl;


    std::vector<std::string> search_params;
	SearchComparator sc = SearchComparator(dim, &kdtree, data);

    for (int i = 0; i < num_search; ++i)
    {
    	if (!std::getline(std::cin, line)){
    		std::cout << "wrong input\n";
    		return 0;
    	}
    	line_content.clear();
		std::stringstream line_stream(line);
    	while(std::getline(line_stream, line_element, ' ')) {
    		line_content.push_back(line_element);
    	}

    	if (line_content.empty()){
    		std::cout << "wrong input\n";
    		return 0;
    	}


		// based on input search parameter run script comparing kd search vs naive
		if (line_content[0] == "kNN"){
			search_params = parse_search_params(2, 6, line_content);
			if (!search_params.size()){
				std::cout << "wrong input\n";
				return 0;
			}

			int k = std::stoi(search_params[0]);
			std::string queue = search_params[1];


			VectorXf query = std::vector<float>(dim);
			if (((int)search_params.size() - 3) > 0) {
                for (int j = 0; j < dim; ++j) {
                    query[j] = std::stof(search_params[j + 2]);
                }
            }
            else {
                std::cout << "random query\n";
                query = data[data.size() / 3].p;
            }
			bool pq = true;
			if (queue == "bh"){
				pq = false;
			}

			sc.compare_kNN_search(k, query, pq, visualize);
		}

		else if (line_content[0] == "sph"){
			search_params = parse_search_params(1, 5, line_content);
            if (!search_params.size()){
                std::cout << "wrong input\n";
                return 0;
            }
            float radius = std::stof(search_params[0]);
            VectorXf query = std::vector<float>(dim);

            if (((int)search_params.size() - 2) > 0){
                for (int j = 0; j < dim; ++j) {
                    query[j] = std::stof(search_params[j + 1]);
                }
			}
			else {
                std::cout << "random query\n";
                query = data[data.size() / 3].p;
			}
			sc.compare_range_search_spherical(radius, query, visualize);
		}

		else if (line_content[0] == "rect"){
			search_params = parse_search_params(2, 8, line_content);
			if (!search_params.size()){
				std::cout << "wrong input\n";
				return 0;
			}
			std::vector<float> ranges = std::vector<float> (dim);
			for (int j = 0; j < dim; ++j) {
				ranges[j] = std::stof(search_params[j]);
            }
            VectorXf query = std::vector<float>(dim);
            if (((int)search_params.size() - dim) > 0){
                for (int j = 0; j < dim; ++j) {
                    query[j] = std::stof(search_params[dim + j]);
                }
            }
            else {
                std::cout << "random query\n";
                query = data[data.size() / 3].p;
            }

			sc.compare_range_search_rectangular(ranges, query, visualize);
		}

		else{
			std::cout << "wrong input\n";
			return 0;
		}

    }

    kdtree.destroy_tree();

    // intentionally commented
    // this is how I run the experiments for gettin data for statistical visualizations

//    SearchComparator::experiment_1(num_samples, num_dimensions, 100, k);
//    unsigned int samples_sizes_exp1 [8] = {1000, 5000, 10000, 50000, 100000, 400000, 800000, 1000000};
//    int num_sample_sizes_exp1 = 8;
//    unsigned int max_leaf_points_sizes [9] = {10, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000};
//    int num_mlp = 9;
//    unsigned int dimensions [3] = {2, 3, 4};
//    int num_dim = 3;

//    for (int j = 0; j < num_dim; ++j) {
//        for (int l = 0; l < num_mlp ; ++l) {
//            std::cout << "dim " << dimensions[j] << " mlp" << max_leaf_points_sizes[l]<< std::endl;
//            SearchComparator::experiment_1_build_diff_distributions_fix_dim_mlp_var_N_dparams(dimensions[j], max_leaf_points_sizes[l]);
//        }
//    }


//    unsigned int samples_sizes_exp2 [1] = {1000000};
//    int num_sample_sizes_exp2 = 1;
//
//    for (int i = 0; i < num_dim; ++i) {
//        for (int m = 0; m < num_sample_sizes_exp2; ++m) {
//            std::cout << "dim " << dimensions[i] << " num_samples" << samples_sizes_exp2[m]<< std::endl;
//            SearchComparator::experiment_2_build_diff_distr_fix_N_dim_var_mlp(samples_sizes_exp2[m], dimensions[i]);
//        }
//    }

//     unsigned int samples_sizes_exp4 [1] = {1000000};
//     int num_sample_sizes_exp4 = 1;
//     std::vector<int> ks = {100, 500, 1000, 10000};
//     std::vector<float> radii = {5.0f, 15.0f, 50.0f, 100.0f};
//     std::vector<float> range_sizes1 = {0.0f, 50.0f, 0.0f, 50.0f, 0.0f, 50.0f, 0.0f, 50.0f};
//     std::vector<float> range_sizes2 = {60.0f, 70.0f, 25.0f, 40.0f, 0.0f, 100.0f, -5.0f, 30.0f};
//     std::vector<std::vector<float>> rect_range = {range_sizes1, range_sizes2};
//     bool binomial_heap = false;
//
//     for (int i = 0; i < num_dim; ++i) {
//         for (int m = 0; m < num_sample_sizes_exp4; ++m) {
//                 std::cout << "dim " << dimensions[i] << " num_samples" << samples_sizes_exp4[m]<< std::endl;
//                 // SearchComparator::experiment_4_kNN_fix_N_dim_k_var_mlp(samples_sizes_exp4[m], dimensions[i], ks, radii, rect_range, binomial_heap);
//         }
//     }

    return 0;
}