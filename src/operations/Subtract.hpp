#ifndef SUBTRACT_HPP
#define SUBTRACT_HPP

#include "Operation.hpp"
#include <memory>

namespace nn::operations {
	class Subtract : public Operation, public std::enable_shared_from_this<Subtract> {
	public:
		Subtract() = default;
		Subtract(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
		Subtract(std::shared_ptr<Tensor> a);
		void calculate() override;
		std::shared_ptr<Tensor> operation() override;
		void derivative(bool recursion = false) override;
		void zero_child_grad(bool recursion = false) override;
	};
}

#endif
