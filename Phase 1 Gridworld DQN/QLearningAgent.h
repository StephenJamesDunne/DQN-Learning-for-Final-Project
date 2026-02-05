#pragma once
#include "Position.h"
#include <map>
#include <random>

// QLearningAgent
// Implements tabular Q-learning for the GridWorld game
// The agent learns a Q-table that estimates the expected reward for taking each action in each state
// The agent uses an epsilon-greedy policy to balance exploration and exploitation during training

// Learning rate (alpha): how much the agent updates its Q-values based on new information
// Discount factor (gamma): how much the agent values future rewards compared to immediate rewards
// Exploration rate (epsilon): probability of taking a random action instead of the best known action

class QLearningAgent {
private:
    std::map<std::pair<Position, Action>, double> q_table;
    double alpha;   
    double gamma;   
    double epsilon; 
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_dist;

public:
	// Constructor with default hyperparameters
    QLearningAgent(double learning_rate = 0.1,
        double discount = 0.99,
        double exploration = 0.5);

	// Get Q-value for a given state-action pair (returns 0 if not present)
    double get_q(const Position& state, Action action) const;

	// Choose action using epsilon-greedy policy
    Action choose_action(const Position& state);

	// Updates Q-table based on the observed transition (state, action, reward, next_state)
	// Uses the Bellman equation: newQValue = currentQValue + alpha(reward + gamma * maxNextQValue - currentQValue)
    void update(const Position& state, Action action,
        double reward, const Position& next_state);

	// Decay the exploration rate (epsilon) after each episode to reduce exploration over time
	// Ideally should default to exploitation (epsilon=0) after enough episodes
    void decay_epsilon(double decay_rate = 0.995, double min_epsilon = 0.01);

	// Print the learned policy (best action at each position) in a grid format
    void print_policy(int grid_size) const;

	// Print the Q-values for all state-action pairs in a readable format
    void print_q_values(int grid_size) const;

    double get_epsilon() const { return epsilon; }
    void set_epsilon(double new_epsilon) { epsilon = new_epsilon; }
};