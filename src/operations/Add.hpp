#ifndef ADD_HPP
#define ADD_HPP

#include "Operation.hpp"
#include <memory>

namespace nn::operations {
	class Add : public Operation, public std::enable_shared_from_this<Add> {
	public:
		Add() = default;
		Add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
		Add(std::shared_ptr<Tensor> a);
		void calculate() override;
		std::shared_ptr<Tensor> operation() override;
		void derivative(bool recursion = false) override;
		void zero_child_grad(bool recursion = false) override;
	};
}

#endif
