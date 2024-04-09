#ifndef MIN_MAX_SACLER_H
#define MIN_MAX_SACLER_H

#include "Preprocessing.hpp"
#include <vector>

namespace nn::preprocessing {
	class MinMaxScaler : public Scaler {
		std::vector<double> m_mins;
		std::vector<double> m_maxs;
	public:
		MinMaxScaler() = default;
		void fit(std::vector<std::shared_ptr<Tensor>> &data) override;
		void transform(std::vector<std::shared_ptr<Tensor>> &data) override;
		void fit_transform(std::vector<std::shared_ptr<Tensor>> &data) override;
	};

};

#endif
