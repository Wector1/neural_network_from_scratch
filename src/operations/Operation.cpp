#include "Operation.hpp"

namespace nn::operations {
	Operation::Operation(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) 
		: a_(a), b_(b) {
		a_->add_reference();
		b_->add_reference();
		
		int size_of_a = 1, size_of_b = 1;
		for (auto& dim : a_->get_dimensions())
			size_of_a *= dim;
		for (auto& dim : b_->get_dimensions())
			size_of_b *= dim;
		if (size_of_a > size_of_b) {
			c_ = std::make_shared<Tensor>(Tensor(a_->get_dimensions()));
		}
		else {
			c_ = std::make_shared<Tensor>(Tensor(b_->get_dimensions()));
		}
	};

	Operation::Operation(std::shared_ptr<Tensor> a)
		: a_(a) {
		a_->add_reference();
		c_ = std::make_shared<Tensor>(Tensor(a_->get_dimensions()));
	};

	void Operation::apply_grad(int batch_size, double learning_rate, bool recursion) {
		a_->add_reference_calculated_();
		if (b_ != nullptr)
			b_->add_reference_calculated_();

		if (recursion) {
			if (a_->all_references_calculated_())
				a_->apply_grad(batch_size, recursion, learning_rate);
			if (b_ != nullptr && b_->all_references_calculated_())
				b_->apply_grad(batch_size, recursion, learning_rate);
		}
	}
}
