cmake_minimum_required (VERSION 3.25)
project(ai
    LANGUAGES CXX
	VERSION 1.0.0
)

add_executable(${PROJECT_NAME}
	examples/square_function.cpp
    src/Tensor.cpp
    src/operations/Operation.cpp
    src/operations/Add.cpp
    src/operations/Multiply.cpp
    src/operations/Subtract.cpp
    src/operations/Divide.cpp
    src/operations/Matrix_mul.cpp
    src/operations/ReLU.cpp
    src/operations/Tanh.cpp
    src/operations/Reduce_sum.cpp
    src/operations/Exp.cpp
    src/operations/Log.cpp
    src/operations/Mse.cpp
	src/operations/Cross_entropy.cpp
    src/operations/Softmax.cpp
    src/layers/Mse.cpp
	src/layers/Cross_entropy.cpp
    src/layers/Softmax.cpp
    src/layers/Linear.cpp
    src/layers/Tanh.cpp
    src/layers/ReLU.cpp
    src/layers/Layer.cpp
    src/models/MLPRegressor.cpp
    src/models/MLPClassifier.cpp
    src/initialization/Initialization.cpp
    src/preprocessing/Preprocessing.hpp
	src/preprocessing/Min_max_scaler.cpp
	src/preprocessing/Standard_scaler.cpp
    src/functions/Activation_factory.cpp
    src/utils/data.cpp
)
