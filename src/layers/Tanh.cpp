#include "Tanh.hpp"
#include "../operations/Tanh.hpp"
#include <iostream>

namespace nn::layer {
	Tanh::Tanh(std::shared_ptr<Tensor> input)
		: Layer(input)
	{
		output_ = operations::tanh(input);
	}

	std::shared_ptr<Tensor> Tanh::forward() {
		output_->forward();
		return output_;
	}

	int Tanh::get_size() {
		return output_->get_size();
	}
}
