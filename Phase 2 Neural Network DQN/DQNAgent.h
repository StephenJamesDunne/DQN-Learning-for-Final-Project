#pragma once
#include "Position.h"
#include "NeuralNetwork.h"
#include "ReplayBuffer.h"
#include <vector>
#include <random>

// Deep Q-Network Agent for Gridworld
// Implements a Deep Q-Network agent that learns to play Gridworld
class DQNAgent {
private:
	NeuralNetwork q_network;        // Main Q-network that learns to predict Q-values for state-action pairs
	NeuralNetwork target_network;   // Target network used to compute stable target Q-values during training (updated less frequently)
	ReplayBuffer replay_buffer;     // Experience replay buffer that stores past experiences for training the Q-network

	double learning_rate;			// Alpha: How much to update Q-values based on new information
	double gamma;					// Discount factor: How much to value future rewards compared to immediate rewards
	double epsilon;					// Exploration rate: Probability of choosing a random action vs. the best action from Q-network
	double epsilon_decay;			// How much to decay epsilon after each episode (multiplier)
	double epsilon_min;				// Minimum epsilon value to ensure some exploration continues

	int batch_size;					// Number of experiences to sample from replay buffer for each training step
	int target_update_frequency;    // How many steps to take before updating the target network with the online network's weights
	int steps_since_target_update;  // Counter to track steps for target network updates
	int total_steps;				// Total steps taken (used for target network updates)

	std::mt19937 rng;										// Random number generator for action selection and replay buffer sampling
	std::uniform_real_distribution<double> uniform_dist;	// For epsilon-greedy action selection

	// Convert state and goal positions into input vector for neural network
    std::vector<double> stateToInput(const Position& state, const Position& goal) const;

public:
	// Constructor
    DQNAgent(double lr = 0.001,
        double discount = 0.99,
        double exploration = 1.0);

	// Choose action using epsilon-greedy policy
    Action chooseAction(const Position& state, const Position& goal);

	// Store experience in replay buffer
    void remember(const Position& state, Action action, double reward,
        const Position& next_state, bool done, const Position& goal);

	// Train the Q-network using a batch of experiences from the replay buffer
    void trainStep();

    void incrementStep();

	// decay epsilon after each episode to reduce exploration over time
    void decayEpsilon();

    double getEpsilon() const { return epsilon; }
    void setEpsilon(double new_epsilon) { epsilon = new_epsilon; }

    size_t getBufferSize() const { return replay_buffer.size(); }
};