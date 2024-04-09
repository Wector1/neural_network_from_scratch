#ifndef CROSS_ENTROPY_LAYER_HPP
#define CROSS_ENTROPY_LAYER_HPP

#include "../Tensor.hpp"
#include "Layer.hpp"

namespace nn::layer {
	class Cross_entropy : public Layer {
		std::shared_ptr<Tensor> correct_output_;
	public:
		Cross_entropy(std::shared_ptr<Tensor> input);
		std::shared_ptr<Tensor> forward() override;
		std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> y);
		std::shared_ptr<Tensor> get_output() { return output_; }
		double get_error();
		int get_size() override;
		void zero_grad();
		std::shared_ptr<Tensor> get_input() override { return input_; }
	};
}

#endif
