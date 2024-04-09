#include "ReLU.hpp"
#include "../operations/ReLU.hpp"
#include <iostream>

namespace nn::layer {
	ReLU::ReLU(std::shared_ptr<Tensor> input)
		: Layer(input)
	{
		output_ = operations::relu(input);
	}

	std::shared_ptr<Tensor> ReLU::forward() {
		output_->forward();
		return output_;
	}

	int ReLU::get_size() {
		return output_->get_size();
	}
}
