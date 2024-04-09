#ifndef OPERATION_HPP
#define OPERATION_HPP
#include <memory>
#include "../Tensor.hpp"

namespace nn::operations {
	class Operation {
	protected:
		std::shared_ptr<Tensor> a_;
		std::shared_ptr<Tensor> b_;
		std::shared_ptr<Tensor> c_;
	public:
		Operation() = default;
		Operation(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
		Operation(std::shared_ptr<Tensor> a);
		virtual void calculate() = 0;
		virtual std::shared_ptr<Tensor> operation() = 0;
		virtual void derivative(bool recursion = false) = 0;
		virtual void zero_child_grad(bool recursion = false) = 0;
		virtual void apply_grad(int batch_size, double learning_rate, bool recursion = false);
	};
}

#endif
