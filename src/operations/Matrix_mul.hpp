#ifndef MATRIX_MUL_HPP
#define MATRIX_MUL_HPP

#include "Operation.hpp"
#include <memory>

namespace nn::operations {
	class Matrix_mul : public Operation, public std::enable_shared_from_this<Matrix_mul> {
	public:
		Matrix_mul() = default;
		Matrix_mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
		Matrix_mul(std::shared_ptr<Tensor> a);
		void calculate() override;
		std::shared_ptr<Tensor> operation() override;
		void derivative(bool recursion = false) override;
		void zero_child_grad(bool recursion = false) override;
	};

	std::shared_ptr<Tensor> matrix_mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
}

#endif
