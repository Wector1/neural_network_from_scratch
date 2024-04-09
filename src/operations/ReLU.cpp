#include "ReLU.hpp"
#include <algorithm>
#include <memory>

namespace nn::operations {
	ReLU::ReLU(std::shared_ptr<Tensor> a)
		: Operation(a) { };


	void ReLU::calculate() {
		for (int i = 0; i < a_->get_size(); i++) {
			double a_val = std::max((*a_)[i], 0.0);
			(*c_)[i] = a_val;		
		}
	}

	std::shared_ptr<Tensor> ReLU::operation() {
		calculate();
		return c_;
	}

	void ReLU::derivative(bool recursion) {
		for (int i = 0; i < a_->get_size(); i++) {
			double c_val = (*c_)[i];
			double grad_c = c_->get_iteration_grad(i);
			a_->add_to_delta_grads(i, (c_val > 0.0 ? 1.0 : 0.0) * grad_c);
			a_->add_to_iteration_grads(i, (c_val > 0.0 ? 1.0 : 0.0) * grad_c);
		}

		a_->add_reference_calculated_();

		if (recursion) {
			if (a_->all_references_calculated_())
				a_->backprop(recursion);
		}
	}

	void ReLU::zero_child_grad(bool recursion) {
		a_->add_reference_calculated_();

		if (recursion) {
			if (a_->all_references_calculated_())
				a_->zero_grad(recursion);
		}
	}

	std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a) {
		std::shared_ptr<ReLU> op = std::make_shared<ReLU>(ReLU(a));
		auto tensor = op->operation();
		tensor->set_operation(op);
		return tensor;
	}
}
