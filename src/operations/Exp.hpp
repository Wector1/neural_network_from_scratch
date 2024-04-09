#ifndef EXP_HPP
#define EXP_HPP

#include "Operation.hpp"
#include <memory>

namespace nn::operations {
	class Exp : public Operation, public std::enable_shared_from_this<Exp> {
	public:
		Exp() = default;
		Exp(std::shared_ptr<Tensor> a);
		void calculate() override;
		std::shared_ptr<Tensor> operation() override;
		void derivative(bool recursion = false) override;
		void zero_child_grad(bool recursion = false) override;
	};

	std::shared_ptr<Tensor> exp(std::shared_ptr<Tensor> a);
}

#endif
