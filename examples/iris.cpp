#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>
#include "../src/Tensor.hpp"
#include "../src/utils/data.hpp"
#include "../src/models/MLPClassifier.hpp"
#include "../src/preprocessing/Standard_scaler.hpp"

using namespace nn;

std::vector<std::string> split_line(std::string line) {
	std::vector<std::string>   result;
    std::stringstream lineStream(line);
    std::string cell;

    while(std::getline(lineStream,cell, ',')) {
        result.push_back(cell);
    }

    return result;
}

void read_data(std::string path,
		std::vector<std::shared_ptr<Tensor>> &X,
		std::vector<std::shared_ptr<Tensor>> &y,
		std::map<std::string, int> &labels) {
    std::fstream file;
	file.open(path);

	std::vector<std::vector<std::string>> data;
	std::string line;
	std::set<std::string> str_labels;
	while (getline(file, line)) {
		if (line == "")
			break;
        data.push_back(split_line(line));
		str_labels.insert(data.back().back());
	}

	int index = 2;
	for (auto &it : str_labels) {
        if (!labels.count(it)) {
			labels[it] = index--;
		}
	}

	for (auto &row : data) {
		double x[4];
		for (int i = 0; i < 4; i++) {
			x[i] = stod(row[i]);
		}
        X.push_back(std::move(
					std::make_shared<Tensor>(Tensor(x, 4))));
		X.back()->reshape({4, 1});
		double labels_ohe[3]{};
		labels_ohe[labels[row[4]]] = 1.0;
		y.push_back(std::move(
					std::make_shared<Tensor>(Tensor(labels_ohe, 3))));
		y.back()->reshape({3, 1});
	}
	file.close();
}

void print_data(std::vector<std::shared_ptr<Tensor>> &X,
		std::vector<std::shared_ptr<Tensor>> &y) {
	for (int i = 0; i < X.size(); i++) {
		std::cout << "data: [";
		for (int j = 0; j < X[i]->get_size(); j++) {
			std::cout << (*X[i])[j] << ", ";
		}
		std::cout << "]" << std::endl;
		std::cout << "labels: [";
		for (int j = 0; j < y[i]->get_size(); j++) {
			std::cout << (*y[i])[j] << ", ";
		}
		std::cout << "]" << std::endl;

	}
}

int main() {
	std::vector<std::shared_ptr<Tensor>> X;
	std::vector<std::shared_ptr<Tensor>> y;
	std::map<std::string, int> labels;
	
	read_data("../data/iris.data", X, y, labels);
	preprocessing::StandardScaler scaler;
	scaler.fit_transform(X);

	std::vector<std::shared_ptr<Tensor>> X_train;
	std::vector<std::shared_ptr<Tensor>> y_train;
	std::vector<std::shared_ptr<Tensor>> X_test;
	std::vector<std::shared_ptr<Tensor>> y_test;
	utils::data::train_test_split(0.1, X, y, X_train, y_train, X_test, y_test);


    auto classifier = models::MLPClassifier(
			4,
			{50, 40, 3},
			1,
			"tanh",
			0.1,
			300,
			true,
			0.0001,
			true,
			10,
			42);

	classifier.fit(X_train, y_train);
	int correct{};
    for (int i = 0; i < X_test.size(); i++) {
	    int output = classifier.predict(X_test[i]);
		int correct_output = 0;
		for (int j = 0; j < 3; j++) {
			if ((*y_test[i])[j] == 1.0)
				correct_output = j;
		}
		correct += output == correct_output;
		std::cout << "predicted: " << output 
			<< ", correct: " << correct_output
			<< (output == correct_output ? " WELL DONE" : " WRONG")<< std::endl;
	}
	std::cout << "accuracy: " << (double)correct / X_test.size() * 100 << '%' << std::endl;

	return 0;
}
