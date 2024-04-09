#include "Tanh.hpp"
#include <cmath>
#include <memory>

namespace nn::operations {
	Tanh::Tanh(std::shared_ptr<Tensor> a)
		: Operation(a) { };


	void Tanh::calculate() {
		for (int i = 0; i < a_->get_size(); i++) {
			double a_val = (*a_)[i];
			(*c_)[i] = (std::exp(a_val) - std::exp(-a_val)) / 
				(std::exp(a_val) + std::exp(-a_val));
		}
	}

	std::shared_ptr<Tensor> Tanh::operation() {
		calculate();
		return c_;
	}

	void Tanh::derivative(bool recursion) {
		for (int i = 0; i < a_->get_size(); i++) {
			double c_val = (*c_)[i];
			double grad_c = c_->get_iteration_grad(i);
			a_->add_to_delta_grads(i, (1 - c_val * c_val) * grad_c);
			a_->add_to_iteration_grads(i, (1 - c_val * c_val) * grad_c);
		}

		a_->add_reference_calculated_();

		if (recursion) {
			if (a_->all_references_calculated_())
				a_->backprop(recursion);
		}
	}

	void Tanh::zero_child_grad(bool recursion) {
		a_->add_reference_calculated_();

		if (recursion) {
			if (a_->all_references_calculated_())
				a_->zero_grad(recursion);
		}
	}

	std::shared_ptr<Tensor> tanh(std::shared_ptr<Tensor> a) {
		std::shared_ptr<Tanh> op = std::make_shared<Tanh>(Tanh(a));
		auto tensor = op->operation();
		tensor->set_operation(op);
		return tensor;
	}
}
