#ifndef TANH_HPP
#define TANH_HPP

#include "Operation.hpp"
#include <memory>

namespace nn::operations {
	class Tanh : public Operation, public std::enable_shared_from_this<Tanh> {
	public:
		Tanh() = default;
		Tanh(std::shared_ptr<Tensor> a);
		void calculate() override;
		std::shared_ptr<Tensor> operation() override;
		void derivative(bool recursion = false) override;
		void zero_child_grad(bool recursion = false) override;
	};

	std::shared_ptr<Tensor> tanh(std::shared_ptr<Tensor> a);
}

#endif
