#include "Log.hpp"
#include <cmath>
#include <memory>

namespace nn::operations {
	Log::Log(std::shared_ptr<Tensor> a)
		: Operation(a) { };


	void Log::calculate() {
		for (int i = 0; i < a_->get_size(); i++) {
			double a_val = std::log((*a_)[i]);
			(*c_)[i] = a_val;
		}
	}

	std::shared_ptr<Tensor> Log::operation() {
		calculate();
		return c_;
	}

	void Log::derivative(bool recursion) {
		for (int i = 0; i < a_->get_size(); i++) {
			double a_val = (*a_)[i];
			double grad_c = c_->get_iteration_grad(i);
			a_->add_to_delta_grads(i, 1 / a_val * grad_c);
			a_->add_to_iteration_grads(i, 1 / a_val * grad_c);
		}

		a_->add_reference_calculated_();

		if (recursion) {
			if (a_->all_references_calculated_())
				a_->backprop(recursion);
		}
	}

	void Log::zero_child_grad(bool recursion) {
		a_->add_reference_calculated_();

		if (recursion) {
			if (a_->all_references_calculated_())
				a_->zero_grad(recursion);
		}
	}

	std::shared_ptr<Tensor> log(std::shared_ptr<Tensor> a) {
		std::shared_ptr<Log> op = std::make_shared<Log>(Log(a));
		auto tensor = op->operation();
		tensor->set_operation(op);
		return tensor;
	}
}
