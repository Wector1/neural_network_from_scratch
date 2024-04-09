#include "Reduce_sum.hpp"
#include <memory>

namespace nn::operations {
	Reduce_sum::Reduce_sum(std::shared_ptr<Tensor> a) {
		c_ = std::make_shared<Tensor>(Tensor(std::vector<int>{1}));
		a_ = a;
		a_->add_reference();
	}


	void Reduce_sum::calculate() {
		double sum{};
		for (int i = 0; i < a_->get_size(); i++) {
			sum += (*a_)[i];
		}
		(*c_)[0] = sum;
	}

	std::shared_ptr<Tensor> Reduce_sum::operation() {
		calculate();
		return c_;
	}

	void Reduce_sum::derivative(bool recursion) {
		for (int i = 0; i < a_->get_size(); i++) {
			double grad_c = c_->get_iteration_grad(0);
			a_->add_to_delta_grads(i, 1.0 * grad_c);
			a_->add_to_iteration_grads(i, 1.0 * grad_c);
		}

		a_->add_reference_calculated_();

		if (recursion) {
			if (a_->all_references_calculated_())
				a_->backprop(recursion);
		}
	}

	void Reduce_sum::zero_child_grad(bool recursion) {
		a_->add_reference_calculated_();

		if (recursion) {
			if (a_->all_references_calculated_())
				a_->zero_grad(recursion);
		}
	}

	std::shared_ptr<Tensor> reduce_sum(std::shared_ptr<Tensor> a) {
		std::shared_ptr<Reduce_sum> op = std::make_shared<Reduce_sum>(Reduce_sum(a));
		auto tensor = op->operation();
		tensor->set_operation(op);
		return tensor;
	}
}
