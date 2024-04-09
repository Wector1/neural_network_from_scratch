#include "Softmax.hpp"
#include "Reduce_sum.hpp"
#include <iostream>
#include <memory>
#include "Exp.hpp"

namespace nn::operations {
	Softmax::Softmax(std::shared_ptr<Tensor> x) {
		a_ = x;
		exp_ = exp(a_);
		reduce_ = reduce_sum(exp_);
		res_ = exp_ / reduce_;
		c_ = std::make_shared<Tensor>(Tensor(std::vector<int>{x->get_size()}));
	}


	void Softmax::calculate() {
        exp_->forward();
        reduce_->forward();
		res_->forward();
        for (int i = 0; i < (*c_).get_size(); i++) {
			(*c_)[i] = (*res_)[i];
		}
	}

	std::shared_ptr<Tensor> Softmax::operation() {
		calculate();
		return c_;
	}

	void Softmax::derivative(bool recursion) {
        for (int i = 0; i < (*c_).get_size(); i++) {
			res_->set_iteration_grads(i, c_->get_iteration_grad(i));
		}
        res_->backprop(recursion);
	}

	void Softmax::zero_child_grad(bool recursion) {
		if (c_->all_references_calculated_()) {
			res_->zero_grad(recursion);
		}
	}

	void Softmax::apply_grad(int batch_size, double learning_rate, bool recursion) {
        reduce_->apply_grad(batch_size, recursion, learning_rate);
	}

	void Softmax::print() {
		std::cout << "- a -\n";
		a_->print();
		a_->print_grad();
		a_->print_iteration_grad();
		std::cout << "- c -\n";
		c_->print();
		c_->print_grad();
		c_->print_iteration_grad();
	}

	std::shared_ptr<Tensor> softmax(std::shared_ptr<Tensor> x) {
		std::shared_ptr<Softmax> op = std::make_shared<Softmax>(Softmax(x));
		auto tensor = op->operation();
		tensor->set_operation(op);
		return tensor;
	}
}
