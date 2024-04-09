#ifndef MLPCLASSIFIER_HPP
#define MLPCLASSIFIER_HPP
#include "../layers/Layer.hpp"
#include "../operations/Operation.hpp"
#include <vector>
#include <random>

namespace nn::models {
	class MLPClassifier {
		std::shared_ptr<Tensor> input_;
		std::mt19937_64 rng;
		std::vector<std::shared_ptr<layer::Layer>> layers_;
		void shuffle(std::vector<std::shared_ptr<Tensor>> &inputs,
				std::vector<std::shared_ptr<Tensor>> &outputs);
		int batch_size_;
		std::string activation_;
		double learning_rate_;
		int max_iter_;
		bool shuffle_;
		double tol_;
		bool verbose_;
		int n_iter_no_change_;
		int seed_;
		
	public:
		MLPClassifier(int number_of_inputs,
				std::vector<int> hidden_layer_sizes,
				int batch_size,
				std::string activation = "tanh",
				double learning_rate = 0.01,
				int max_iter = 50,
				bool shuffle = true,
				double tol = 0.0001,
				bool verbose = true,
				int n_iter_no_change = 10,
				int seed = 42);
		void fit(std::vector<std::shared_ptr<Tensor>> X,
				std::vector<std::shared_ptr<Tensor>> y);

		int predict(std::shared_ptr<Tensor> X);
	};
}

#endif
