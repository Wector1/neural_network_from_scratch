#ifndef MSE_HPP
#define MSE_HPP

#include "Operation.hpp"
#include <memory>

namespace nn::operations {
	class Mse : public Operation, public std::enable_shared_from_this<Mse> {
	private:
		std::shared_ptr<Tensor> difference_;
		std::shared_ptr<Tensor> squared_;
		std::shared_ptr<Tensor> reduce_;
	public:
		Mse() = default;
		Mse(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y);
		void calculate() override;
		std::shared_ptr<Tensor> get_output() { return reduce_; }
		std::shared_ptr<Tensor> operation() override;
		void derivative(bool recursion = false) override;
		void zero_child_grad(bool recursion = false) override;
		void apply_grad(int batch_size, double learning_rate, bool recursion = false) override;
		void print();
	};

	std::shared_ptr<Tensor> mse(std::shared_ptr<Tensor> x, 
			std::shared_ptr<Tensor> y);
}

#endif
