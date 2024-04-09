#ifndef SOFTMAX_LAYER_HPP
#define SOFTMAX_LAYER_HPP

#include "../Tensor.hpp"
#include "Layer.hpp"

namespace nn::layer {
	class Softmax : public Layer {
	public:
		Softmax(std::shared_ptr<Tensor> input);
		std::shared_ptr<Tensor> forward() override;
		int get_size() override;
		std::shared_ptr<Tensor> get_input() override { return input_; }
	};
}

#endif
