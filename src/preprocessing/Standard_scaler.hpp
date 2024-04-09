#ifndef STANDARD_SCALER_H
#define STANDARD_SCALER_H

#include "Preprocessing.hpp"
#include <vector>
#include "../Tensor.hpp"

namespace nn::preprocessing {
	class StandardScaler : Scaler {
		std::vector<double> m_means;
		std::vector<double> m_stdevs;
	public:
		void fit(std::vector<std::shared_ptr<Tensor>> &data) override;
		void transform(std::vector<std::shared_ptr<Tensor>> &data) override;
		void fit_transform(std::vector<std::shared_ptr<Tensor>> &data) override;
	};

};

#endif
