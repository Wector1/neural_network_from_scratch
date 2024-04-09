#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include "Standard_scaler.hpp"

namespace nn::preprocessing {
	void StandardScaler::fit(std::vector<std::shared_ptr<Tensor>> &data) {
		for (int i = 0; i < data[0]->get_size(); i++) {
			std::vector<double> column(data.size());
			for (int j = 0; j < data.size(); j++) {
				column[j] = (*data[j])[i];
			}
			m_means.emplace_back(std::reduce(column.begin(), column.end()) / column.size()); 

			std::vector<double> diff(column.size());
			std::transform(column.begin(), column.end(), diff.begin(),
					[&](double x) { return x - m_means.back(); });
			double sqSum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
			m_stdevs.emplace_back(std::sqrt(sqSum / column.size()));
		}
	}

	void StandardScaler::transform(std::vector<std::shared_ptr<Tensor>> &data) {
		for (int i = 0; i < data[0]->get_size(); i++) {
			for (int j = 0; j < data.size(); j++) {
				(*data[j])[i] = ((*data[j])[i] - m_means[i]) / m_stdevs[i];
			}
		}
	}

	void StandardScaler::fit_transform(std::vector<std::shared_ptr<Tensor>> &data) {
		fit(data);
		transform(data);
	}
}
