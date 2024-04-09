#ifndef LINEAR_HPP
#define LINEAR_HPP

#include "../Tensor.hpp"
#include "Layer.hpp"

namespace nn::layer {
	class Linear : public Layer {
		std::shared_ptr<Tensor> weights_;
		std::shared_ptr<Tensor> matr_mul_;
		std::shared_ptr<Tensor> biases_;
	public:
		Linear(std::shared_ptr<Tensor> input, int output_size);
		void set_input(std::shared_ptr<Tensor> input);
		std::shared_ptr<Tensor> forward() override;
		std::shared_ptr<Tensor> get_weights() { return weights_; }
		std::shared_ptr<Tensor> get_biases() { return biases_; }
		int get_size() override;
		std::shared_ptr<Tensor> get_input() override { return input_; }
	};
}

#endif
