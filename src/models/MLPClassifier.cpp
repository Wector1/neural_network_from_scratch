#include "MLPClassifier.hpp"
#include "../layers/Linear.hpp"
#include "../operations/Cross_entropy.hpp"
#include "../layers/Softmax.hpp"
#include "../layers/Cross_entropy.hpp"
#include "../initialization/Initialization.hpp"
#include "../functions/Activation_factory.hpp"
#include <cmath>
#include <vector>
#include <iostream>

namespace nn::models {
	MLPClassifier::MLPClassifier(int number_of_inputs,
			std::vector<int> hidden_layer_sizes,
			int batch_size,
			std::string activation,
			double learning_rate,
			int max_iter,
			bool shuffle,
			double tol,
			bool verbose,
			int n_iter_no_change,
			int seed) :
		batch_size_(batch_size),
		activation_(activation),
		learning_rate_(learning_rate),
		max_iter_(max_iter),
		shuffle_(shuffle),
		tol_(tol),
		verbose_(verbose),
		n_iter_no_change_(n_iter_no_change),
		seed_(seed) {

		Initialization::rng.seed(seed_);
		std::shared_ptr<Tensor> input_ = std::make_shared<Tensor>(
				Tensor(std::vector<int>{number_of_inputs}));
		input_->reshape({number_of_inputs, 1});

		if (hidden_layer_sizes.size() > 0) {
			layers_.push_back(std::move(std::make_shared<layer::Linear>(
							layer::Linear(input_, hidden_layer_sizes[0]))));
			layers_.push_back(std::move(functions::create_activation_layer(
							activation, layers_.back())));
		}

		for (int i = 1; i < hidden_layer_sizes.size(); i++) {
			layers_.push_back(std::move(std::make_shared<layer::Linear>(
							layer::Linear(layers_.back()->forward(),
								hidden_layer_sizes[i]))));
			if (i == hidden_layer_sizes.size() - 1) {
				if (activation == "relu") {
					layers_.push_back(std::move(functions::create_activation_layer(
									"tanh", layers_.back())));
				} else {
					layers_.push_back(std::move(functions::create_activation_layer(
									activation, layers_.back())));
				}
			} else {
				layers_.push_back(std::move(functions::create_activation_layer(
								activation, layers_.back())));
			}
		}
		layers_.push_back(std::move(std::make_shared<layer::Softmax>(
						layer::Softmax(layers_.back()->forward()))));
		layers_.push_back(std::move(std::make_shared<layer::Cross_entropy>(
						layer::Cross_entropy(layers_.back()->forward()))));
	}

	void MLPClassifier::shuffle(std::vector<std::shared_ptr<Tensor>> &inputs,
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

	void MLPClassifier::fit(std::vector<std::shared_ptr<Tensor>> X,
			std::vector<std::shared_ptr<Tensor>> y) {
		rng.seed(seed_);
		double best_error = INFINITY;
		int best_error_iter = 0;
		for (int i = 0; i < max_iter_; i++) {
			int correctly_classified_this_epoch{};
			if (shuffle_)
				shuffle(X, y);
			double epoch_error{};

			for (int j = 0; j < (int)X.size(); j += batch_size_) {
				double batch_error{};
				for (int k = j; k < j + batch_size_ && k < (int)X.size(); k++) {
					std::dynamic_pointer_cast<layer::Linear>(layers_[0])->set_input(X[k]);
					std::dynamic_pointer_cast<layer::Cross_entropy>(layers_.back())->zero_grad();

					std::shared_ptr<Tensor> res;
					for (int l = 0; l < (int)layers_.size() - 1; l++) {
						res = layers_[l]->forward();
					}

					if (verbose_) {
						int predicted{};
						for (int i = 0; i < res->get_size(); i++) {
							if ((*res)[predicted] < (*res)[i])
								predicted = i;
						}
						int correct{};
						for (int i = 0; i < y[k]->get_size(); i++) {
							if ((*y[k])[i] == 1.0)
								correct = i;
						}
						correctly_classified_this_epoch += (correct == predicted);
					}

					auto error = std::dynamic_pointer_cast<layer::Cross_entropy>(layers_.back())->forward(y[k]);

					error->backprop(true);
					batch_error += (*error)[0];
					epoch_error += (*error)[0];
				}
				auto output = std::dynamic_pointer_cast<layer::Cross_entropy>(layers_.back())->get_output();
				output->apply_grad(batch_size_, true, learning_rate_);
			}

			if (verbose_) {
				std::cout << "--- epoch " << i + 1 << " ---" << std::endl;
				std::cout << "accuracy on training set: " << 
					(double)correctly_classified_this_epoch / X.size() * 100 << "%, ";
				std::cout << " error: " << epoch_error / X.size() << std::endl;
			}

			if (epoch_error - best_error < -tol_) {
				best_error = epoch_error;
				best_error_iter = i;
			} else if (i - best_error_iter > n_iter_no_change_) {
				return;
			}
		}
		std::cout << "Warning: Convergence not reached yet" << std::endl;
	}

	int MLPClassifier::predict(std::shared_ptr<Tensor> X) {

		std::dynamic_pointer_cast<layer::Linear>(layers_[0])->set_input(X);
		std::shared_ptr<Tensor> res;
		for (int l = 0; l < (int)layers_.size() - 1; l++) {
			res = layers_[l]->forward();
		}
		int max_idx{};
		for (int i = 0; i < res->get_size(); i++) {
			if ((*res)[i] > (*res)[max_idx])
				max_idx = i;
		}

		return max_idx;
	}
}
