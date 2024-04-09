#include "Mse.hpp"
#include "../operations/Mse.hpp"
#include <iostream>
#include <memory>

namespace nn::layer {
	Mse::Mse(std::shared_ptr<Tensor> input)
		: Layer(input)
	{
		correct_output_ = std::make_shared<Tensor>(Tensor(input->get_dimensions()));
		output_ = operations::mse(input, correct_output_);
	}

	double Mse::get_error() {
		return (*output_)[0];
	}

	std::shared_ptr<Tensor> Mse::forward() {
		output_->forward();
		return output_;
	}

	int Mse::get_size() {
		return output_->get_size();
	}

	void Mse::zero_grad() {
		output_->zero_grad(true);
	}

	std::shared_ptr<Tensor> Mse::forward(std::shared_ptr<Tensor> y) {
		*correct_output_ = *y;
		output_->forward();
		return output_;
	}
}
