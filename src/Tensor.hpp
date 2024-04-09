#ifndef TENSOR_HPP
#define TENSOR_HPP
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace nn {
	namespace operations {
		class Operation;
    }

	class Tensor {
		std::string label_;
		std::vector<double> tensor_;
		std::vector<double> delta_grads_;
		std::vector<double> iteration_grads_;
		std::vector<int> dimensions_;
		std::shared_ptr<operations::Operation> operation_;
		std::shared_ptr<Tensor> a_;
		std::shared_ptr<Tensor> b_;
		void print_tensor(std::vector<int> dim, int curr_dim);
		void print_grad(std::vector<int> dims, int curr_dim);
		void print_iteration_grad(std::vector<int> dims, int curr_dim);
		int references_ = 0;
		int references_calculated_ = 0;
		bool is_trainable_ = false;
	//	double learning_rate_ = 0.05;
	public:
		Tensor(double array[], int size, std::string label = "");
		Tensor(std::vector<double> array, bool trainable = false);
		Tensor(double array[], int size, std::unique_ptr<operations::Operation> operation, 
				std::string label = "");
		Tensor(std::vector<int> dimensions, bool trainable = false);
		double get_delta_grad(int idx);
		double get_iteration_grad(int idx);
		void set_iteration_grads(double value);
		void set_iteration_grads(int idx, double value);
		void set_delta_grads(int idx, double value);
		void apply_grad(int batch_size, bool recursion = false, double learning_rate = 0.05);
		void add_to_delta_grads(int idx, double value);
		void add_to_iteration_grads(int idx, double value);
		void add_to_delta_grads(double value);
		void add_reference();
		void subtract_reference();
		void zero_references();
		void add_reference_calculated_();
		bool all_references_calculated_();
		void zero_references_calculated_();
		void zero_grad(bool recursion = false);
		void backprop(bool recursion = false);
		std::shared_ptr<Tensor> forward();
		void mul();
		void set_operation(std::shared_ptr<operations::Operation> operation);
		void set_label(std::string label);
		void set_trainable(bool is_trainable);
		bool is_trainable() { return is_trainable_; }
		void print();
		void print_grad();
		void print_iteration_grad();
		void print_info();
		std::vector<int> get_dimensions() { return dimensions_; }
		std::shared_ptr<operations::Operation> get_operation() { return operation_; }
		double * get_tensor() { return tensor_.data(); }
		std::vector<double> get_tensor_vec() { return tensor_; }
		int get_size() { return (int)tensor_.size(); }
		void reshape(std::vector<int> shapes);
		double operator[](int idx) const {
			return tensor_[idx];
		}
		double& operator[](int idx) {
			return tensor_[idx];
		}

		std::shared_ptr<Tensor> operator-();
	};

	std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
	std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
	std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
	std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
	std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
}

#endif
