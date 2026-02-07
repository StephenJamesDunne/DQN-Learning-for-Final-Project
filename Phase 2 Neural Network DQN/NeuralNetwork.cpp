#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const std::vector<int>& layers)
    : layer_sizes(layers) {
    std::random_device rd;
    rng = std::mt19937(rd());
    initializeWeights();
}

void NeuralNetwork::initializeWeights() {
    weights.clear();
    biases.clear();

    // Initialize weights for each layer
    for (size_t layer = 0; layer < layer_sizes.size() - 1; layer++) {
        int input_size = layer_sizes[layer];
        int output_size = layer_sizes[layer + 1];

        // Xavier initialization: sqrt(2 / (input_size + output_size))
        double stddev = std::sqrt(2.0 / (input_size + output_size));
        std::normal_distribution<double> dist(0.0, stddev);

        // Create weight matrix for this layer
        std::vector<std::vector<double>> layer_weights;
        for (int out = 0; out < output_size; out++) {
            std::vector<double> neuron_weights;
            for (int in = 0; in < input_size; in++) {
                neuron_weights.push_back(dist(rng));
            }
            layer_weights.push_back(neuron_weights);
        }
        weights.push_back(layer_weights);

        // Initialize biases to zero
        std::vector<double> layer_biases(output_size, 0.0);
        biases.push_back(layer_biases);
    }
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    activations.clear();
    z_values.clear();

    // Store input as first activation
    activations.push_back(input);

    std::vector<double> current_activation = input;

    // Process each layer
    for (size_t layer = 0; layer < weights.size(); layer++) {
        std::vector<double> z;
        std::vector<double> next_activation;

        // For each neuron in current layer
        for (size_t neuron = 0; neuron < weights[layer].size(); neuron++) {
            // Weighted sum + bias
            double sum = biases[layer][neuron];

            for (size_t prev = 0; prev < current_activation.size(); prev++) {
                sum += weights[layer][neuron][prev] * current_activation[prev];
            }

            z.push_back(sum);

            // Apply activation function
            if (layer < weights.size() - 1) {
                // Hidden layers: ReLU
                next_activation.push_back(relu(sum));
            }
            else {
                // Output layer: Linear (no activation)
                next_activation.push_back(sum);
            }
        }

        z_values.push_back(z);
        activations.push_back(next_activation);
        current_activation = next_activation;
    }

    return current_activation;
}

void NeuralNetwork::backward(const std::vector<double>& input,
    int action,
    double td_error,
    double learning_rate) {

	// Step 1: Compute output layer gradients
    std::vector<double> output_gradients(layer_sizes.back(), 0.0);
    output_gradients[action] = td_error;  // Only chosen action has error

	// Step 2: Backpropagate gradients through hidden layers
    std::vector<std::vector<double>> all_gradients;
    all_gradients.push_back(output_gradients);

    // Go backwards through layers
    for (int layer = static_cast<int>(weights.size()) - 2; layer >= 0; layer--) {
        std::vector<double> layer_gradients(layer_sizes[layer + 1], 0.0);

        for (size_t neuron = 0; neuron < layer_sizes[layer + 1]; neuron++) {
            double gradient = 0.0;

            // Sum gradients from next layer
            for (size_t next_neuron = 0; next_neuron < all_gradients[0].size(); next_neuron++) {
                gradient += all_gradients[0][next_neuron] *
                    weights[layer + 1][next_neuron][neuron];
            }

            // Apply ReLU derivative
            if (z_values[layer][neuron] > 0) {
                layer_gradients[neuron] = gradient;
            }
            else {
                layer_gradients[neuron] = 0.0;  // Dead neuron
            }
        }

        all_gradients.insert(all_gradients.begin(), layer_gradients);
    }

	// Step 3: Update weights and biases using computed gradients
    for (size_t layer = 0; layer < weights.size(); layer++) {
        for (size_t neuron = 0; neuron < weights[layer].size(); neuron++) {
            // Update weights
            for (size_t prev = 0; prev < weights[layer][neuron].size(); prev++) {
                double gradient = all_gradients[layer][neuron] * activations[layer][prev];
                weights[layer][neuron][prev] += learning_rate * gradient;
            }

            // Update bias
            biases[layer][neuron] += learning_rate * all_gradients[layer][neuron];
        }
    }
}

// Need to copy weights from online network to target network periodically to stabilize training
void NeuralNetwork::copyWeightsFrom(const NeuralNetwork& other) {
    this->weights = other.weights;
    this->biases = other.biases;
}