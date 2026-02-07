#include "DQNAgent.h"
#include <algorithm>
#include <iostream>

DQNAgent::DQNAgent(double lr, double discount, double exploration)
    : q_network({ 4, 16, 16, 4 }),
    target_network({ 4, 16, 16, 4 }),
    replay_buffer(10000),
    learning_rate(lr),
    gamma(discount),
    epsilon(exploration),
    epsilon_decay(0.995),
    epsilon_min(0.01),
    batch_size(32),
    target_update_frequency(100),
    steps_since_target_update(0),
	total_steps(0),
    uniform_dist(0.0, 1.0)
{
    std::random_device rd;
    rng = std::mt19937(rd());

    // Initialize target network with same weights as Q-network
    target_network.copyWeightsFrom(q_network);
}

std::vector<double> DQNAgent::stateToInput(const Position& state, const Position& goal) const {
    // Normalize positions to [0, 1]
    return {
        state.x / 8.0,
        state.y / 8.0,
        goal.x / 8.0,
        goal.y / 8.0
    };
}

Action DQNAgent::chooseAction(const Position& state, const Position& goal) {
    // Exploration: random action
    if (uniform_dist(rng) < epsilon) {
        return static_cast<Action>(rng() % 4);
    }

    // Exploitation: choose best action from Q-network
    std::vector<double> input = stateToInput(state, goal);
    std::vector<double> q_values = q_network.forward(input);

    // Find action with max Q-value
    int best_action = 0;
    double max_q = q_values[0];

    for (int a = 1; a < 4; a++) {
        if (q_values[a] > max_q) {
            max_q = q_values[a];
            best_action = a;
        }
    }

    return static_cast<Action>(best_action);
}

void DQNAgent::remember(const Position& state, Action action, double reward,
    const Position& next_state, bool done, const Position& goal) {
    Experience exp;
    exp.state = stateToInput(state, goal);
    exp.action = static_cast<int>(action);
    exp.reward = reward;
    exp.next_state = stateToInput(next_state, goal);
    exp.done = done;

    replay_buffer.add(exp);
}

void DQNAgent::trainStep() {
    // Don't train if not enough experiences
    if (!replay_buffer.canSample(batch_size)) {
        return;
    }

    // Sample random batch from replay buffer
    const std::vector<Experience>& batch = replay_buffer.sample(batch_size);

    // Train on each experience in the batch
    for (const auto& exp : batch) {
        // Compute target Q-value using target network
        std::vector<double> next_q_values = target_network.forward(exp.next_state);
        double max_next_q = *std::max_element(next_q_values.begin(), next_q_values.end());

        // Bellman equation
        double target_q = exp.done ? exp.reward : exp.reward + gamma * max_next_q;

        // Current Q-value from online network
        std::vector<double> current_q_values = q_network.forward(exp.state);
        double current_q = current_q_values[exp.action];

        // TD error
        double td_error = target_q - current_q;

        // Backpropagate through Q-network
        q_network.backward(exp.state, exp.action, td_error, learning_rate);
    }
}

void DQNAgent::incrementStep() {
    total_steps++;

    if (total_steps % target_update_frequency == 0) {
        target_network.copyWeightsFrom(q_network);
        
        // Only print every 1000 steps to avoid flooding console
        if (total_steps % 1000 == 0) {
            std::cout << "  [Target network updated at step " << total_steps << "]\n";
        }
    }
}

void DQNAgent::decayEpsilon() {
    epsilon = std::max(epsilon_min, epsilon * epsilon_decay);
}