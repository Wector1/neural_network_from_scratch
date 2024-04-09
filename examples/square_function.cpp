#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <random>
#include "../src/Tensor.hpp"
#include "../src/models/MLPRegressor.hpp"
#include "../src/preprocessing/Min_max_scaler.hpp"
#include "../src/utils/data.hpp"

using namespace nn;

int main() {
	std::mt19937_64 rng;
    auto regressor = models::MLPRegressor(
			1, 
			{20, 10, 2},
			16,
			"tanh",
			0.1,
			50,
			true,
			0.0001,
			true,
			10,
			100);
	std::vector<std::shared_ptr<Tensor>> inputs;
    std::vector<std::shared_ptr<Tensor>> outputs;

	std::normal_distribution<double> normal_distribution(0, 30.0);
	for (int i = -1000; i < 1000; i += 1) {
		std::cout << normal_distribution(rng) << std::endl;
		double y = (double)i * i + normal_distribution(rng) * i;
		inputs.push_back(std::move(std::make_shared<Tensor>(
						Tensor(std::vector<double>(
								{(double)i})))));
		inputs.back()->reshape({1, 1});
		outputs.push_back(std::move(std::make_shared<Tensor>(
						Tensor(std::vector<double>({y})))));
		outputs.back()->reshape({1, 1});
	}

	auto inputScaler = preprocessing::MinMaxScaler();
	auto outputScaler = preprocessing::MinMaxScaler();
	inputScaler.fit_transform(inputs);
 	outputScaler.fit_transform(outputs);

	utils::data::rng.seed(42);

	std::vector<std::shared_ptr<Tensor>> X_train;
    std::vector<std::shared_ptr<Tensor>> y_train;
	std::vector<std::shared_ptr<Tensor>> X_test;
    std::vector<std::shared_ptr<Tensor>> y_test;

	utils::data::train_test_split(0.2, inputs, outputs,
		X_train, y_train, X_test, y_test);


	regressor.fit(X_train, y_train);
    std::ofstream file;
	file.open("../debugs/function.csv", std::ios_base::out);
	file << "num,prediction,real\n";


	for (int i = 0; i < X_test.size(); i++) {
		auto output = regressor.predict(X_test[i])[0];
		double correct = (*y_test[i])[0];
		file << (*X_test[i])[0] << ',' << output << ',' << correct << std::endl;
		std::cout << "input: "
			<< (*X_test[i])[0]
			<< ", predicted: " << output
			<< ", correct: " << correct
			<< std::endl;
	}
}
