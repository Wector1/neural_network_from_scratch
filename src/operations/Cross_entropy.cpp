#include "Cross_entropy.hpp"
#include "Reduce_sum.hpp"
#include "Log.hpp"
#include <iostream>
#include <memory>

namespace nn::operations {
	Cross_entropy::Cross_entropy(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y) {
		a_ = x;
		b_ = y;
		log_ = log(a_);
		mul_ = log_ * b_;
		reduce_ = reduce_sum(mul_);
		c_ = std::make_shared<Tensor>(Tensor(std::vector<int>{1}));
	}


	void Cross_entropy::calculate() {
		log_->forward();
		mul_->forward();
		reduce_->forward();
		(*c_)[0] = -(*reduce_)[0];
	}

	std::shared_ptr<Tensor> Cross_entropy::operation() {
		calculate();
		return c_;
	}

	void Cross_entropy::derivative(bool recursion) {
		reduce_->set_iteration_grads(-1.0);
		reduce_->backprop(recursion);
	}

	void Cross_entropy::zero_child_grad(bool recursion) {
		reduce_->zero_grad(recursion);
	}

	void Cross_entropy::apply_grad(int batch_size, double learning_rate, bool recursion) {
		reduce_->apply_grad(batch_size, recursion, learning_rate);
	}

	void Cross_entropy::print() {
		std::cout << "- a -\n";
		a_->print();
		a_->print_grad();
		a_->print_iteration_grad();
		std::cout << "- b -\n";
		b_->print();
		b_->print_grad();
		b_->print_iteration_grad();
		std::cout << "- c -\n";
		c_->print();
		c_->print_grad();
		c_->print_iteration_grad();
	}

	std::shared_ptr<Tensor> cross_entropy(std::shared_ptr<Tensor> x, 
			std::shared_ptr<Tensor> y) {
		std::shared_ptr<Cross_entropy> op = std::make_shared<Cross_entropy>(Cross_entropy(x, y));
		auto tensor = op->operation();
		tensor->set_operation(op);
		return tensor;
	}
}
