#include "Activation_factory.hpp"
#include "../initialization/Initialization.hpp"
#include "../layers/Linear.hpp"
#include "../layers/Tanh.hpp"
#include "../layers/ReLU.hpp"
#include <stdexcept>

namespace nn::functions {
	std::shared_ptr<layer::Layer> create_activation_layer(
			std::string function_name,
			std::shared_ptr<layer::Layer> layer) {
		std::shared_ptr<layer::Layer> activation;
		for (auto &c : function_name)
			c = std::tolower(c);

		if (function_name == "tanh") {
			Initialization::xavier(std::dynamic_pointer_cast<layer::Linear>
					(layer)->get_weights(), layer->get_input()->get_size());
			Initialization::xavier(std::dynamic_pointer_cast<layer::Linear>
					(layer)->get_biases(), layer->get_input()->get_size());
			activation = std::make_shared<layer::Tanh>(
							layer::Tanh(layer->forward()));
		} else if (function_name == "relu") {
			Initialization::he(std::dynamic_pointer_cast<layer::Linear>
					(layer)->get_weights(), layer->get_input()->get_size());
			Initialization::he(std::dynamic_pointer_cast<layer::Linear>
					(layer)->get_biases(), layer->get_input()->get_size());
			activation = std::make_shared<layer::ReLU>(
							layer::ReLU(layer->forward()));

		} else if (function_name == "sigmoid") {

		} else {
			throw std::invalid_argument(
					"there is no activation function named: " + function_name);
		}

		return activation;
	}
}
