#ifndef LOG_HPP
#define LOG_HPP

#include "Operation.hpp"
#include <memory>

namespace nn::operations {
	class Log : public Operation, public std::enable_shared_from_this<Log> {
	public:
		Log() = default;
		Log(std::shared_ptr<Tensor> a);
		void calculate() override;
		std::shared_ptr<Tensor> operation() override;
		void derivative(bool recursion = false) override;
		void zero_child_grad(bool recursion = false) override;
	};

	std::shared_ptr<Tensor> log(std::shared_ptr<Tensor> a);
}

#endif
