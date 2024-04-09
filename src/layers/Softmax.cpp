#include "Softmax.hpp"
#include "../operations/Softmax.hpp"

namespace nn::layer {
	Softmax::Softmax(std::shared_ptr<Tensor> input)
		: Layer(input)
	{
		output_ = operations::softmax(input);
	}

	std::shared_ptr<Tensor> Softmax::forward() {
		output_->forward();
		return output_;
	}

	int Softmax::get_size() {
		return output_->get_size();
	}
}
