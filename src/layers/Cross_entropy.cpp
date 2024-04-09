#include "Cross_entropy.hpp"
#include "../operations/Cross_entropy.hpp"
#include <memory>

namespace nn::layer {
	Cross_entropy::Cross_entropy(std::shared_ptr<Tensor> input)
		: Layer(input)
	{
		correct_output_ = std::make_shared<Tensor>(Tensor(input->get_dimensions()));
		output_ = operations::cross_entropy(input, correct_output_);
	}

	double Cross_entropy::get_error() {
		return (*output_)[0];
	}

	std::shared_ptr<Tensor> Cross_entropy::forward() {
		output_->forward();
		return output_;
	}

	int Cross_entropy::get_size() {
		return output_->get_size();
	}

	void Cross_entropy::zero_grad() {
		output_->zero_grad(true);
	}
	std::shared_ptr<Tensor> Cross_entropy::forward(std::shared_ptr<Tensor> y) {
		*correct_output_ = *y;
		output_->forward();
		return output_;
	}
}
