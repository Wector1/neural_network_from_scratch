#include "Initialization.hpp"
#include <random>

namespace nn::Initialization {
	std::mt19937_64 rng;
    void xavier(std::shared_ptr<Tensor> weights,
		int size_of_previous_layer) {
		std::uniform_real_distribution<double> unif((double)-1 /
				sqrt((double)size_of_previous_layer), (double)1 /
				sqrt(double(size_of_previous_layer)));
		for (int i = 0; i < weights->get_size(); i++) {
			(*weights)[i] = unif(rng);
		}
	}

    void he(std::shared_ptr<Tensor> weights,
		int size_of_previous_layer) {
		std::normal_distribution<double> normal(0.0, 
				sqrt((double) 2 / size_of_previous_layer));
		for (int i = 0; i < weights->get_size(); i++) {
			(*weights)[i] = normal(rng);
		}
	}
}

