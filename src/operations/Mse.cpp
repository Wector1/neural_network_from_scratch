#include "Mse.hpp"
#include "Reduce_sum.hpp"
#include <iostream>
#include <memory>

namespace nn::operations {
	Mse::Mse(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y) {
		a_ = x;
		b_ = y;
		difference_ = b_ - a_;
		squared_ = difference_ * difference_;
		reduce_ = reduce_sum(squared_);
		c_ = std::make_shared<Tensor>(Tensor(std::vector<int>{1}));
	}


	void Mse::calculate() {
        difference_->forward();
        squared_->forward();
        reduce_->forward();
		(*c_)[0] = (*reduce_)[0];
	}

	std::shared_ptr<Tensor> Mse::operation() {
		calculate();
		return c_;
	}

	void Mse::derivative(bool recursion) {
		reduce_->set_iteration_grads(1.0);
		reduce_->backprop(recursion);
	}

	void Mse::zero_child_grad(bool recursion) {
        reduce_->zero_grad(recursion);
	}

	void Mse::apply_grad(int batch_size, double learning_rate, bool recursion) {
        reduce_->apply_grad(batch_size, recursion, learning_rate);
	}

	void Mse::print() {
		std::cout << "- a -\n";
		a_->print();
		a_->print_grad();
		a_->print_iteration_grad();
		std::cout << "- b -\n";
		b_->print();
		b_->print_grad();
		b_->print_iteration_grad();
		std::cout << "- difference -\n";
		difference_->print();
		difference_->print_grad();
		difference_->print_iteration_grad();
		std::cout << "- squared -\n";
		squared_->print();
		squared_->print_grad();
		squared_->print_iteration_grad();
		std::cout << "- c -\n";
		c_->print();
		c_->print_grad();
		c_->print_iteration_grad();
	}

	std::shared_ptr<Tensor> mse(std::shared_ptr<Tensor> x, 
			std::shared_ptr<Tensor> y) {
		std::shared_ptr<Mse> op = std::make_shared<Mse>(Mse(x, y));
		auto tensor = op->operation();
		tensor->set_operation(op);
		return tensor;
	}
}
