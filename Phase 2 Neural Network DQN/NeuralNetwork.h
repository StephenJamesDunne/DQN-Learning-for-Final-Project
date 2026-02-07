#pragma once
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

// Simple feedforward neural network for approximating Q-values
// Architecture: Input layer (state representation) -> Hidden layers (ReLU) -> Output layer (Q-values for each action)
// For the 8x8 gridworld:
// Input layer has 64 neurons
// Output layer has 4 neurons (Q-values for UP, RIGHT, DOWN, LEFT)
class NeuralNetwork {
private:
    // Network architecture
    std::vector<int> layer_sizes;

    // Weights[layer][neuron_out][neuron_in]
    std::vector<std::vector<std::vector<double>>> weights;

    // Biases[layer][neuron]
    std::vector<std::vector<double>> biases;

    // Activations stored during forward pass (needed for backpropagation)
    std::vector<std::vector<double>> activations;
    std::vector<std::vector<double>> z_values;  // Pre-activation values

    std::mt19937 rng;

    // Activation functions
    double relu(double x) const {
        return std::max(0.0, x);
    }

    double relu_derivative(double x) const {
        return x > 0 ? 1.0 : 0.0;
    }

public:
    // Constructor
	// Initialize the network architecture and random number generator
    explicit NeuralNetwork(const std::vector<int>& layers);

	// Use Xavier initialization for weights and zero initialization for biases
    void initializeWeights();

	// Forward pass: compute Q-values for a given input state
    std::vector<double> forward(const std::vector<double>& input);

	// Backward pass: update weights based on TD-error (target value - predicted Q-value)
    void backward(const std::vector<double>& input,
        int action,
        double td_error,
        double learning_rate);

	// Copy weights from another network (used for target network updates)
    void copyWeightsFrom(const NeuralNetwork& other);

    size_t getLayerCount() const { return layer_sizes.size(); }
};