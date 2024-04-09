#include "Matrix_mul.hpp"
#include <memory>

namespace nn::operations {
	Matrix_mul::Matrix_mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
		a_ = a;
		b_ = b;
		a_->add_reference();
		b_->add_reference();
		auto shape_a = a_->get_dimensions();
		auto shape_b = b_->get_dimensions();
		c_ = std::make_shared<Tensor>(Tensor(
					std::vector<int>{
						shape_a[shape_a.size() - 2],
						shape_b.back()
					}
				));
	}

	void Matrix_mul::calculate() {
		// available only for 2d tensors
		//
		auto tensor_a = a_->get_tensor(); 
		auto tensor_b = b_->get_tensor(); 
		auto shape_a = a_->get_dimensions();
		auto shape_b = b_->get_dimensions();
		int cols_a = shape_a.back();
		int rows_a = shape_a[shape_a.size() - 2];
		int cols_b = shape_b.back();
		for (int i = 0; i < c_->get_size(); i++) {
			(*c_)[i] = 0;
		}
		
		for (int i = 0; i < cols_b; i++) {
			for (int j = 0; j < rows_a; j++) {
				for (int k = 0; k < cols_a; k++) {
					(*c_)[j * cols_b + i] += (*a_)[j * cols_a + k]
						* (*b_)[k * cols_b + i];
				}
			}
		}
	}
	std::shared_ptr<Tensor> Matrix_mul::operation() {
		calculate();
		return c_;
	}

	void Matrix_mul::derivative(bool recursion) {

		auto tensor_a = a_->get_tensor(); 
		auto tensor_b = b_->get_tensor(); 
		auto shape_a = a_->get_dimensions();
		auto shape_b = b_->get_dimensions();
		int cols_a = shape_a.back();
		int rows_a = shape_a[shape_a.size() - 2];
		int cols_b = shape_b.back();
		
		for (int i = 0; i < cols_b; i++) {
			for (int j = 0; j < rows_a; j++) {
				for (int k = 0; k < cols_a; k++) {
					double b_val = (*b_)[k * cols_b + i];
					double a_val = (*a_)[j * cols_a + k];
					double grad_c = c_->get_iteration_grad(j * cols_b + i);

					a_->add_to_delta_grads(j * cols_a + k, b_val * grad_c);
					b_->add_to_delta_grads(k * cols_b + i, a_val * grad_c);
					a_->add_to_iteration_grads(j * cols_a + k, b_val * grad_c);
					b_->add_to_iteration_grads(k * cols_b + i, a_val * grad_c);
				}
			}
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

	void Matrix_mul::zero_child_grad(bool recursion) {
		a_->add_reference_calculated_();
		b_->add_reference_calculated_();

		if (recursion) {
			if (a_->all_references_calculated_())
				a_->zero_grad(recursion);
			if (b_->all_references_calculated_())
				b_->zero_grad(recursion);
		}
	}


	std::shared_ptr<Tensor> matrix_mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
		std::shared_ptr<Matrix_mul> op = std::make_shared<Matrix_mul>(Matrix_mul(a, b));
		auto tensor = op->operation();
		tensor->set_operation(op);
		return tensor;
	}
}
