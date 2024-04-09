#ifndef INITIALIZATION_HPP
#define INITIALIZATION_HPP

#include "../Tensor.hpp"
#include <random>

namespace nn::Initialization {
    extern std::mt19937_64 rng;
    void xavier(std::shared_ptr<Tensor> weights,
			int size_of_previous_layer);
    void he(std::shared_ptr<Tensor> weights,
			int size_of_previous_layer);

}

#endif
