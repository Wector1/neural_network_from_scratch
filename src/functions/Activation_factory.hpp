#ifndef ACTIVATION_FACTORY_HPP
#define ACTIVATION_FACTORY_HPP

#include <memory>
#include "../operations/Operation.hpp"
#include "../layers/Layer.hpp"

namespace nn::functions {
	std::shared_ptr<layer::Layer> create_activation_layer(
			std::string function_name,
			std::shared_ptr<layer::Layer> layer);
}

#endif
