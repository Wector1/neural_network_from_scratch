#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <vector>
#include "../Tensor.hpp"

namespace nn::preprocessing {
    class Scaler {
	public:
		Scaler() = default;
		virtual void fit(std::vector<std::shared_ptr<Tensor>> &data) = 0;
		virtual void transform(std::vector<std::shared_ptr<Tensor>> &data) = 0;
		virtual void fit_transform(std::vector<std::shared_ptr<Tensor>> &data) = 0;
	};
}

#endif
