#include "Add.hpp"
#include <memory>

namespace nn::operations {
	Add::Add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b)
		: Operation(a, b) {	};

	void Add::calculate() {
		auto tensor_a = a_->get_tensor(); 
		auto tensor_b = b_->get_tensor(); 

		int size_of_a = a_->get_size(),	size_of_b = b_->get_size();

		for (int i = 0; i < std::max(size_of_a, size_of_b); i++) {
			(*c_)[i] = tensor_a[i % size_of_a] + tensor_b[i % size_of_b];
		}
	}
	std::shared_ptr<Tensor> Add::operation() {
		calculate();
		return c_;
	}

	void Add::derivative(bool recursion) {
		auto tensor_a = a_->get_tensor(); 
		auto tensor_b = b_->get_tensor(); 
		auto tensor_c = c_->get_tensor(); 

		int size_of_a = a_->get_size(),	size_of_b = b_->get_size();

		for (int i = 0; i < std::max(size_of_a, size_of_b); i++) {
			double grad_c = c_->get_iteration_grad(i);
			a_->add_to_delta_grads(i, 1 * grad_c);
			b_->add_to_delta_grads(i, 1 * grad_c);
			a_->add_to_iteration_grads(i, 1 * grad_c);
			b_->add_to_iteration_grads(i, 1 * grad_c);
		}

		a_->add_reference_calculated_();
		b_->add_reference_calculated_();

		if (recursion) {
			if (a_->all_references_calculated_())
				a_->backprop(recursion);
			if (b_->all_references_calculated_())
				b_->backprop(recursion);
		}
	}

	void Add::zero_child_grad(bool recursion) {
		a_->add_reference_calculated_();
		b_->add_reference_calculated_();

		if (recursion) {
			if (a_->all_references_calculated_())
				a_->zero_grad(recursion);
			if (b_->all_references_calculated_())
				b_->zero_grad(recursion);
		}
	}
}
