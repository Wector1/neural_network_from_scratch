#ifndef REDUCE_SUM_HPP
#define REDUCE_SUM_HPP

#include "Operation.hpp"
#include <memory>

namespace nn::operations {
	class Reduce_sum : public Operation, public std::enable_shared_from_this<Reduce_sum> {
	public:
		Reduce_sum() = default;
		Reduce_sum(std::shared_ptr<Tensor> a);
		void calculate() override;
		std::shared_ptr<Tensor> operation() override;
		void derivative(bool recursion = false) override;
		void zero_child_grad(bool recursion = false) override;
	};

	std::shared_ptr<Tensor> reduce_sum(std::shared_ptr<Tensor> a);
}

#endif
