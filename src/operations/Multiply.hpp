#ifndef MULTIPLY_HPP
#define MULTIPLY_HPP
#include "Operation.hpp"
#include <memory>

namespace nn::operations {
	class Multiply : public Operation, public std::enable_shared_from_this<Multiply> {
	public:
		Multiply() = default;
		Multiply(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
		Multiply(std::shared_ptr<Tensor> a);
		void calculate() override;
		std::shared_ptr<Tensor> operation() override;
		void derivative(bool recursion = false) override;
		void zero_child_grad(bool recursion = false) override;
	};
}

#endif
