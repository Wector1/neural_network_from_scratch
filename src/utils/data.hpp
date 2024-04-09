#ifndef UTILS_DATA_HPP
#define UTILS_DATA_HPP

#include <random>
#include <vector>
#include "../Tensor.hpp"

namespace nn::utils::data{
    extern std::mt19937_64 rng;
	void shuffle(std::vector<std::shared_ptr<Tensor>> &inputs,
			std::vector<std::shared_ptr<Tensor>> &outputs);
	void train_test_split(double test_size,
		std::vector<std::shared_ptr<Tensor>> inputs,
		std::vector<std::shared_ptr<Tensor>> outputs,
		std::vector<std::shared_ptr<Tensor>> &train_inputs,
		std::vector<std::shared_ptr<Tensor>> &train_outputs,
		std::vector<std::shared_ptr<Tensor>> &test_inputs,
		std::vector<std::shared_ptr<Tensor>> &test_outputs,
		bool to_shuffle = true
	);
}

#endif
