#include "Layer.hpp"

namespace nn::layer {
	Layer::Layer(std::shared_ptr<Tensor> input) {
		input_ = input;
	}
}
