#include "data.hpp"
#include <numeric>

namespace nn::utils::data {
	std::mt19937_64 rng;
	void shuffle(std::vector<std::shared_ptr<Tensor>> &inputs,
			std::vector<std::shared_ptr<Tensor>> &outputs) {
		std::vector<int> indexes(inputs.size());
		std::iota(indexes.begin(), indexes.end(), 0);
		std::shuffle(indexes.begin(), indexes.end(), rng);
		std::vector<std::shared_ptr<Tensor>> inputs_shuffled;
		std::vector<std::shared_ptr<Tensor>> outputs_shuffled;

		for (auto &idx : indexes) {
            inputs_shuffled.emplace_back(inputs[idx]);
            outputs_shuffled.emplace_back(outputs[idx]);
		}
		inputs = std::move(inputs_shuffled);
		outputs = std::move(outputs_shuffled);
	}
	void train_test_split(double test_size,
		std::vector<std::shared_ptr<Tensor>> inputs,
		std::vector<std::shared_ptr<Tensor>> outputs,
		std::vector<std::shared_ptr<Tensor>> &train_inputs,
		std::vector<std::shared_ptr<Tensor>> &train_outputs,
		std::vector<std::shared_ptr<Tensor>> &test_inputs,
		std::vector<std::shared_ptr<Tensor>> &test_outputs,
		bool to_shuffle
	) {
        if (to_shuffle) {
			shuffle (inputs, outputs);
		}

		for (int i = 0; i < inputs.size() * test_size; i++) {
			test_inputs.push_back(inputs[i]);
			test_outputs.push_back(outputs[i]);
		}

		for (int i = inputs.size() * test_size; i < inputs.size(); i++) {
			train_inputs.push_back(inputs[i]);
			train_outputs.push_back(outputs[i]);
		}
	}
}
