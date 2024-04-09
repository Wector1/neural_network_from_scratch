#include <algorithm>
#include <vector>
#include "Min_max_scaler.hpp"

namespace nn::preprocessing {
	void MinMaxScaler::fit(std::vector<std::shared_ptr<Tensor>> &data) {
		for (int i = 0; i < data[0]->get_size(); i++) {
			m_mins.push_back((*data[0])[i]);
			m_maxs.push_back((*data[0])[i]);
			for (int j = 0; j < data.size(); j++) {
				m_mins[i] = std::min((*data[j])[i], m_mins[i]);
				m_maxs[i] = std::max((*data[j])[i], m_maxs[i]);
			}
		}
	}

	void MinMaxScaler::transform(std::vector<std::shared_ptr<Tensor>> &data) {
		for (int i = 0; i < data[0]->get_size(); i++) {
			for (int j = 0; j < data.size(); j++) {
				(*data[j])[i] = ((*data[j])[i] - m_mins[i]) / ((m_maxs[i] - m_mins[i]) == 0.0 ? 1.0 : m_maxs[i] - m_mins[i]);
			}
		}
	}

	void MinMaxScaler::fit_transform(std::vector<std::shared_ptr<Tensor>> &data) {
		fit(data);
		transform(data);
	}
}
