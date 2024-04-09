#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "Operation.hpp"
#include <memory>

namespace nn::operations {
	class Softmax : public Operation, public std::enable_shared_from_this<Softmax> {
	private:
		std::shared_ptr<Tensor> exp_;
		std::shared_ptr<Tensor> reduce_;
		std::shared_ptr<Tensor> res_;
	public:
		Softmax() = default;
		Softmax(std::shared_ptr<Tensor> x);
		void calculate() override;
		std::shared_ptr<Tensor> operation() override;
		void derivative(bool recursion = false) override;
		void zero_child_grad(bool recursion = false) override;
		void apply_grad(int batch_size, double learning_rate, bool recursion = false) override;
		void print();
	};

	std::shared_ptr<Tensor> softmax(std::shared_ptr<Tensor> x);
}

#endif
