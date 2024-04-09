#include "Tensor.hpp"
#include "operations/Add.hpp"
#include "operations/Divide.hpp"
#include "operations/Multiply.hpp"
#include "operations/Subtract.hpp"
#include "operations/Matrix_mul.hpp"
#include <bits/ranges_algo.h>
#include <iostream>

namespace nn {
	Tensor::Tensor(double array[], int size, std::string label)
		: label_(label) {
		int array_size = size;
		dimensions_ = {array_size};
		tensor_.resize(array_size);
		delta_grads_.resize(array_size);
		iteration_grads_.resize(array_size);
		for (int i = 0; i < array_size; i++) {
			tensor_[i] = array[i];
		}
	}

	Tensor::Tensor(std::vector<double> array, bool trainable) 
		: is_trainable_(trainable) {
		tensor_ = array;
		int array_size = tensor_.size();
		dimensions_ = {(int)tensor_.size()};
		delta_grads_.resize((int)tensor_.size());
		iteration_grads_.resize((int)tensor_.size());
	}

	Tensor::Tensor(double array[], int size, std::unique_ptr<operations::Operation> operation, 
			std::string label)
		: Tensor(array, size, label) {
	}

	Tensor::Tensor(std::vector<int> dimensions, bool trainable)
		: dimensions_(dimensions),
		  is_trainable_(trainable) {
		int number_of_elements = 1;
		for (auto &dim : dimensions_)
			number_of_elements *= dim;
		tensor_.resize(number_of_elements);
		delta_grads_.resize(number_of_elements);
		iteration_grads_.resize(number_of_elements);
	}

	void Tensor::print_tensor(std::vector<int> dims, int curr_dim) {
		if (curr_dim == dims.size() - 1) {
			if (dims.size() != 1 && dims[curr_dim - 1] != 0) {
				for (int i = 0; i < (int)dims.size() - 1; i++) {
					std::cout << ' ';
				}
			}
			std::cout << "[ ";
			int idx = 0;
			int size = tensor_.size();
			for (int i = 0; i < dimensions_.size() - 1; i++) {
				size /= dimensions_[i];
				idx += size * dims[i];
			}
			for (int i = idx; i < idx + dimensions_.back(); i++) {
				std::cout << tensor_[i] << " ";
			}
			if (dims.size() == 1 || dims[curr_dim - 1] == dimensions_[curr_dim - 1] - 1)
				std::cout << " ]";
			else
				std::cout << " ]," << std::endl;
		}
		else {
			std::cout << "[";
			for (int i = 0; i < dimensions_[curr_dim]; i++) {
				dims[curr_dim] = i;
				print_tensor(dims, curr_dim + 1);
			}
			std::cout << "]";
		}
	}

	void Tensor::print_grad(std::vector<int> dims, int curr_dim) {
		if (curr_dim == dims.size() - 1) {
			if (dims.size() != 1 && dims[curr_dim - 1] != 0) {
				for (int i = 0; i < (int)dims.size() - 1; i++) {
					std::cout << ' ';
				}
			}
			std::cout << "[ ";
			int idx = 0;
			int size = tensor_.size();
			for (int i = 0; i < dimensions_.size() - 1; i++) {
				size /= dimensions_[i];
				idx += size * dims[i];
			}
			for (int i = idx; i < idx + dimensions_.back(); i++) {
				std::cout << delta_grads_[i] << " ";
			}
			if (dims.size() == 1 || dims[curr_dim - 1] == dimensions_[curr_dim - 1] - 1)
				std::cout << " ]";
			else
				std::cout << " ]," << std::endl;
		}
		else {
			std::cout << "[";
			for (int i = 0; i < dimensions_[curr_dim]; i++) {
				dims[curr_dim] = i;
				print_grad(dims, curr_dim + 1);
			}
			std::cout << "]";
		}
	}

	void Tensor::print_iteration_grad(std::vector<int> dims, int curr_dim) {
		if (curr_dim == dims.size() - 1) {
			if (dims.size() != 1 && dims[curr_dim - 1] != 0) {
				for (int i = 0; i < (int)dims.size() - 1; i++) {
					std::cout << ' ';
				}
			}
			std::cout << "[ ";
			int idx = 0;
			int size = tensor_.size();
			for (int i = 0; i < dimensions_.size() - 1; i++) {
				size /= dimensions_[i];
				idx += size * dims[i];
			}
			for (int i = idx; i < idx + dimensions_.back(); i++) {
				std::cout << iteration_grads_[i] << " ";
			}
			if (dims.size() == 1 || dims[curr_dim - 1] == dimensions_[curr_dim - 1] - 1)
				std::cout << " ]";
			else
				std::cout << " ]," << std::endl;
		}
		else {
			std::cout << "[";
			for (int i = 0; i < dimensions_[curr_dim]; i++) {
				dims[curr_dim] = i;
				print_iteration_grad(dims, curr_dim + 1);
			}
			std::cout << "]";
		}
	}

	double Tensor::get_delta_grad(int idx) {
		return delta_grads_[idx]; 
	}

	double Tensor::get_iteration_grad(int idx) {
		return iteration_grads_[idx]; 
	}

	void Tensor::set_iteration_grads(double value) {
		for (auto &it : iteration_grads_) {
			it = value;
		}
	}

	void Tensor::set_iteration_grads(int idx, double value) {
		iteration_grads_[idx] = value;
	}
	void Tensor::set_delta_grads(int idx, double value) {
		delta_grads_[idx] = value;
	}

	void Tensor::apply_grad(int batch_size, bool recursion, double learning_rate) {
		if (is_trainable_) {
			for (int i = 0; i < delta_grads_.size(); i++) {
				tensor_[i] -= learning_rate * delta_grads_[i] / batch_size;
			}
		}
		for (auto &it : delta_grads_) {
			it = 0;
		}
		if (recursion && operation_ != nullptr) {
			operation_->apply_grad(batch_size, learning_rate, recursion);
		}
		references_calculated_ = 0;
	}

	void Tensor::add_to_delta_grads(int idx, double value) {
		delta_grads_[idx] += value;
	}

	void Tensor::add_to_iteration_grads(int idx, double value) {
		iteration_grads_[idx] += value;
	}

	void Tensor::add_reference() {
		++references_;
	}

	void Tensor::subtract_reference() {
		--references_;
	}

	void Tensor::zero_references() {
		references_ = 0;
	}

	void Tensor::add_reference_calculated_() {
		++references_calculated_;
	}

	bool Tensor::all_references_calculated_() {
		return references_ == references_calculated_;
	}

	void Tensor::zero_references_calculated_() {
		references_calculated_ = 0;
	}

	void Tensor::zero_grad(bool recursion) {
		for (auto &it : iteration_grads_) {
			it = 0;
		}
		if (recursion && operation_ != nullptr) {
			operation_->zero_child_grad(recursion);
		}
		references_calculated_ = 0;
	}

	void Tensor::backprop(bool recursion) {
		if (recursion && operation_ != nullptr) {
			operation_->derivative(recursion);
		}
		references_calculated_ = 0;
	}

	std::shared_ptr<Tensor> Tensor::forward() {
		return operation_->operation();
	}

	void Tensor::set_operation(std::shared_ptr<operations::Operation> operation) {
		operation_ = std::move(operation);    
	}

	void Tensor::set_label(std::string label) {
		label_ = label;
	}

	void Tensor::set_trainable(bool is_trainable) {
		is_trainable_ = is_trainable;
	}

	void Tensor::print() {
		std::cout << "references: " << references_ << ", references_calc: " << references_calculated_ << std::endl;
		std::cout << "dims: ";
		for (auto &it : dimensions_)
			std::cout << it << ' ';
		std::cout << std::endl;
		std::vector<int> dims(dimensions_.size());
		print_tensor(dims, 0);
		std::cout << std::endl;
	}

	void Tensor::print_grad() {
		std::vector<int> dims(dimensions_.size());
		print_grad(dims, 0);
		std::cout << std::endl;
	}

	void Tensor::print_iteration_grad() {
		std::vector<int> dims(dimensions_.size());
		print_iteration_grad(dims, 0);
		std::cout << std::endl;
	}

	void Tensor::print_info() {

	}

	void Tensor::reshape(std::vector<int> shapes) {
		int new_size = 1;
		for (auto &dim : shapes)
			new_size *= dim;

		if (new_size == tensor_.size()) {
			dimensions_ = std::move(shapes);
		}
	}

	std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
		std::shared_ptr<operations::Matrix_mul> op = 
			std::make_shared<operations::Matrix_mul>(operations::Matrix_mul(a, b));
		auto tensor = op->operation();
		tensor->set_operation(op);
		return tensor;
	}

	std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
		std::shared_ptr<operations::Add> op = 
			std::make_shared<operations::Add>(operations::Add(a, b));
		auto tensor = op->operation();
		tensor->set_operation(op);
		return tensor;
	}

	std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
		std::shared_ptr<operations::Subtract> op = 
			std::make_shared<operations::Subtract>(operations::Subtract(a, b));
		auto tensor = op->operation();
		tensor->set_operation(op);
		return tensor;
	}
		
	std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
		std::shared_ptr<operations::Multiply> op = 
			std::make_shared<operations::Multiply>(operations::Multiply(a, b));
		auto tensor = op->operation();
		tensor->set_operation(op);
		return tensor;
	}

	std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
		std::shared_ptr<operations::Divide> op = 
			std::make_shared<operations::Divide>(operations::Divide(a, b));
		auto tensor = op->operation();
		tensor->set_operation(op);
		return tensor;
	}
}

