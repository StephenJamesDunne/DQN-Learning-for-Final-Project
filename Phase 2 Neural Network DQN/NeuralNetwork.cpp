#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const std::vector<int>& layers)
    : layerSizes(layers) {
    std::random_device rd;
    rng = std::mt19937(rd());
    initializeWeights();
}

void NeuralNetwork::initializeWeights() {
    weights.clear();
    biases.clear();

    // Initialize weights for each layer
    for (size_t layer = 0; layer < layerSizes.size() - 1; layer++) {
        int inputSize = layerSizes[layer];
        int outputSize = layerSizes[layer + 1];

        // Xavier initialization: sqrt(2 / (inputsize + outputsize))
        double stddev = std::sqrt(2.0 / (inputSize + outputSize));

		// Normal distribution for weight initialization
		// Needed to ensure weights are small enough to prevent exploding gradients but large enough to allow learning
        std::normal_distribution<double> dist(0.0, stddev);

        // Create weight matrix for this layer
		// Layer is a collection of neurons, each neuron has weights for all inputs from previous layer
		// 2D vector of size [output_size][input_size] to store weights for each neuron in the layer
        std::vector<std::vector<double>> layerWeights;

        for (int out = 0; out < outputSize; out++) 
        {
            std::vector<double> neuronWeights;

            for (int in = 0; in < inputSize; in++) 
            {
                neuronWeights.push_back(dist(rng));
            }
            layerWeights.push_back(neuronWeights);
        }
        weights.push_back(layerWeights);

        // Initialize biases to zero
        std::vector<double> layerBiases(outputSize, 0.0);
        biases.push_back(layerBiases);
    }
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    activations.clear();
    zValues.clear();

    // Store input as first activation
    activations.push_back(input);

    std::vector<double> currentActivation = input;

    // Process each layer
    for (size_t layer = 0; layer < weights.size(); layer++) 
    {
		// Need two vectors: one for pre-activation values (z) and one for post-activation values (nextActivation)
        std::vector<double> z;
        std::vector<double> nextActivation;

        // For each neuron in current layer
        for (size_t neuron = 0; neuron < weights[layer].size(); neuron++) 
        {
            // Weighted sum + bias
            double sum = biases[layer][neuron];

            for (size_t prev = 0; prev < currentActivation.size(); prev++) 
            {
                sum += weights[layer][neuron][prev] * currentActivation[prev];
            }

            z.push_back(sum);

            // Apply activation function
            if (layer < weights.size() - 1) 
            {
                // Hidden layers: ReLU
                nextActivation.push_back(relu(sum));
            }
            else 
            {
                // Output layer: Linear (no activation)
                nextActivation.push_back(sum);
            }
        }

        zValues.push_back(z);
        activations.push_back(nextActivation);
        currentActivation = nextActivation;
    }

    return currentActivation;
}

// Back propagation to update weights based on TD-error for the chosen action
void NeuralNetwork::backward(const std::vector<double>& input,
    int action,
    double tdError,
    double learningRate) 
{
	// Step 1: Compute output layer gradients
    std::vector<double> outputGradients(layerSizes.back(), 0.0);
    outputGradients[action] = tdError;  // Only chosen action has error

	// Step 2: Backpropagate gradients through hidden layers
    std::vector<std::vector<double>> allGradients;
    allGradients.push_back(outputGradients);

    // Go backwards through layers
    for (int layer = static_cast<int>(weights.size()) - 2; layer >= 0; layer--) 
    {
        std::vector<double> layerGradients(layerSizes[layer + 1], 0.0);

        for (size_t neuron = 0; neuron < layerSizes[layer + 1]; neuron++) 
        {
            double gradient = 0.0;

            // Sum gradients from next layer
            for (size_t next_neuron = 0; next_neuron < allGradients[0].size(); next_neuron++) 
            {
                gradient += allGradients[0][next_neuron] *
                    weights[layer + 1][next_neuron][neuron];
            }

            // Apply ReLU derivative
            if (zValues[layer][neuron] > 0) 
            {
                layerGradients[neuron] = gradient;
            }
            else 
            {
                layerGradients[neuron] = 0.0;  // Dead neuron
            }
        }

		// Insert this layer's gradients at the front of the list (so output layer is last)
        allGradients.insert(allGradients.begin(), layerGradients);
    }

	// Step 3: Update weights and biases using computed gradients
    for (size_t layer = 0; layer < weights.size(); layer++) 
    {
		// Loop through each neuron in the layer and update its weights and bias based on the computed gradient for that neuron
        for (size_t neuron = 0; neuron < weights[layer].size(); neuron++) 
        {
            // Update weights
            for (size_t prev = 0; prev < weights[layer][neuron].size(); prev++) 
            {
                double gradient = allGradients[layer][neuron] * activations[layer][prev];
                weights[layer][neuron][prev] += learningRate * gradient;
            }

            // Update bias
            biases[layer][neuron] += learningRate * allGradients[layer][neuron];
        }
    }
}

// Need to copy weights from online network to target network periodically to stabilize training
void NeuralNetwork::copyWeightsFrom(const NeuralNetwork& other) {
    this->weights = other.weights;
    this->biases = other.biases;
}