#include "QLearningAgent.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

QLearningAgent::QLearningAgent(double learning_rate,
    double discount,
    double exploration)
    : alpha(learning_rate),
    gamma(discount),
    epsilon(exploration),
    uniformDist(0.0, 1.0) {
    std::random_device rd;
    rng = std::mt19937(rd());
}

double QLearningAgent::getQ(const Position& state, Action action) const {
    auto key = std::make_pair(state, action);
    auto it = qTable.find(key);

	// If the state-action pair is not in the Q-table, return 1.0 for optimistic initialization
	// Optimistic initialization: encourages the agent to explore new actions
	// Basically tells agent to assume all actions are good until it learns otherwise, which promotes exploration
    if (it == qTable.end()) {
        return 1.0;
    }

	// If the state-action pair is found in the Q-table, return the stored Q-value
    return it->second;
}

Action QLearningAgent::chooseAction(const Position& state) {
    // Exploration: random action
    if (uniformDist(rng) < epsilon) {
        return static_cast<Action>(rng() % 4);
    }

    // Exploitation: choose best action
    double best_q = getQ(state, UP);
    Action best_action = UP;

	// Check all actions to find the one with the highest Q-value
    for (int a = 1; a < 4; a++) {
        Action action = static_cast<Action>(a);
        double q = getQ(state, action);
        if (q > best_q) {
            best_q = q;
            best_action = action;
        }
    }

    return best_action;
}

void QLearningAgent::update(const Position& state, Action action,
    double reward, const Position& next_state) {
    // Current Q-value
    double current_q = getQ(state, action);

    // Max Q-value for next state
    double max_next_q = getQ(next_state, UP);
    for (int a = 1; a < 4; a++) {
        max_next_q = std::max(max_next_q, getQ(next_state, static_cast<Action>(a)));
    }

    // Bellman equation: newQValue = currentQValue + alpha(reward + gamma * maxNextQValue - currentQValue)
    double new_q = current_q + alpha * (reward + gamma * max_next_q - current_q);

	// Update Q-table with new Q-value for this state-action pair
    qTable[std::make_pair(state, action)] = new_q;
}

// Epsilon needs to decay over time to allow the agent to shift from exploration to exploitation as it learns
// This function reduces epsilon by multiplying it with a decay rate, but ensures it doesn't go below a minimum threshold
void QLearningAgent::decayEpsilon(double decay_rate, double min_epsilon) {
    epsilon = std::max(min_epsilon, epsilon * decay_rate);
}

void QLearningAgent::printPolicy(int grid_size) const {
    std::cout << "\nLearned Policy (best action at each position):\n";

    for (int y = 0; y < grid_size; y++) {
        for (int x = 0; x < grid_size; x++) {
            Position pos{ x, y };
            Action best_action = UP;
            double best_q = getQ(pos, UP);

            // Find best action for this position
            for (int a = 1; a < 4; a++) {
                Action action = static_cast<Action>(a);
                if (getQ(pos, action) > best_q) {
                    best_q = getQ(pos, action);
                    best_action = action;
                }
            }

            // Display as arrow
            char arrow;
            switch (best_action) {
            case UP:    arrow = '^'; break;
            case RIGHT: arrow = '>'; break;
            case DOWN:  arrow = 'v'; break;
            case LEFT:  arrow = '<'; break;
            }

            // Mark goal position
            if (x == grid_size - 1 && y == grid_size - 1) {
                std::cout << "G ";
            }
            else {
                std::cout << arrow << " ";
            }
        }
        std::cout << "\n";
    }
}

void QLearningAgent::printQValues(int grid_size) const {
    std::cout << "\nQ-Values at each position (showing max Q):\n";

    for (int y = 0; y < grid_size; y++) {
        for (int x = 0; x < grid_size; x++) {
            Position pos{ x, y };
            double max_q = getQ(pos, UP);

            // Find max Q-value over all actions
            for (int a = 1; a < 4; a++) {
                max_q = std::max(max_q, getQ(pos, static_cast<Action>(a)));
            }

			// std:: fixed: always show (n) digits after decimal point
			// std:: setprecision(2): show 2 digits after decimal point
			// std:: setw(6): set width of output to 6 characters (for alignment in grid on console)
            std::cout << std::fixed << std::setprecision(2)
                << std::setw(6) << max_q << " ";
        }
        std::cout << "\n";
    }
}