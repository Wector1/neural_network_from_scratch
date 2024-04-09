#include "Linear.hpp"
#include "../operations/Matrix_mul.hpp"

namespace nn::layer {
	Linear::Linear(std::shared_ptr<Tensor> input, int output_size)
		: Layer(input)
	{
		weights_ = std::make_shared<Tensor>(Tensor(
					std::vector<int>{output_size, input->get_size()},
					true));
		biases_ = std::make_shared<Tensor>(
				Tensor(std::vector<int>{output_size, 1},
					true));
		matr_mul_ = operations::matrix_mul(weights_, input);
        output_ = matr_mul_ + biases_;
	}

	void Linear::set_input(std::shared_ptr<Tensor> input) {
		*input_ = *input;
	}

	std::shared_ptr<Tensor> Linear::forward() {
		matr_mul_->forward();
		output_->forward();
		return output_;
	}

	int Linear::get_size() {
		return output_->get_size();
	}
}
