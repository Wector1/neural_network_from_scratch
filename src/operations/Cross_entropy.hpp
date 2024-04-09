#ifndef OPERATION_CROSS_ENTROPY_HPP
#define OPERATION_CROSS_ENTROPY_HPP

#include "Operation.hpp"
#include <memory>

namespace nn::operations {
	class Cross_entropy : public Operation,
		public std::enable_shared_from_this<Cross_entropy> {
	private:
		std::shared_ptr<Tensor> log_;
		std::shared_ptr<Tensor> mul_;
		std::shared_ptr<Tensor> reduce_;
	public:
		Cross_entropy() = default;
		Cross_entropy(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y);
		void calculate() override;
		std::shared_ptr<Tensor> get_output() { return c_; }
		std::shared_ptr<Tensor> operation() override;
		void derivative(bool recursion = false) override;
		void zero_child_grad(bool recursion = false) override;
		void apply_grad(int batch_size, double learning_rate, bool recursion = false) override;
		void print();
	};

	std::shared_ptr<Tensor> cross_entropy(std::shared_ptr<Tensor> x, 
			std::shared_ptr<Tensor> y);
}

#endif
