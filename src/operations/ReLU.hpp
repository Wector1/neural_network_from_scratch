#ifndef RELU_HPP
#define RELU_HPP

#include "Operation.hpp"
#include <memory>

namespace nn::operations {
	class ReLU : public Operation, public std::enable_shared_from_this<ReLU> {
	public:
		ReLU() = default;
		ReLU(std::shared_ptr<Tensor> a);
		void calculate() override;
		std::shared_ptr<Tensor> operation() override;
		void derivative(bool recursion = false) override;
		void zero_child_grad(bool recursion = false) override;
	};

	std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a);
}

#endif
