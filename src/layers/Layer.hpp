#ifndef LAYER_HPP
#define LAYER_HPP

#include "../Tensor.hpp"

namespace nn::layer {
	class Layer {
	protected:
		std::shared_ptr<Tensor> input_;
		std::shared_ptr<Tensor> output_;
	public:
		Layer() = default;
		Layer(std::shared_ptr<Tensor> input, int output_size);
		Layer(std::shared_ptr<Tensor> input);
		Layer(std::shared_ptr<Tensor> input1,
				std::shared_ptr<Tensor> input2);
		virtual std::shared_ptr<Tensor> forward() = 0;
		virtual int get_size() = 0;
		virtual std::shared_ptr<Tensor> get_input() = 0;
		~Layer() = default;
	};
}

#endif
