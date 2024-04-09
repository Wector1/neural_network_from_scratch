#ifndef DIVIDE_HPP
#define DIVIDE_HPP

#include "Operation.hpp"
#include <memory>

namespace nn::operations {
	class Divide : public Operation, public std::enable_shared_from_this<Divide> {
	public:
		Divide() = default;
		Divide(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
		Divide(std::shared_ptr<Tensor> a);
		void calculate() override;
		std::shared_ptr<Tensor> operation() override;
		void derivative(bool recursion = false) override;
		void zero_child_grad(bool recursion = false) override;
	};
}

#endif
