#include "QLearningAgent.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

QLearningAgent::QLearningAgent(double learningRate,
    double discount,
    double exploration)
    : alpha(learningRate),
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
    double bestQ = getQ(state, UP);
    Action bestAction = UP;

	// Check all actions to find the one with the highest Q-value
    for (int a = 1; a < 4; a++) {
        Action action = static_cast<Action>(a);
        double q = getQ(state, action);
        if (q > bestQ) {
            bestQ = q;
            bestAction = action;
        }
    }

    return bestAction;
}

void QLearningAgent::update(const Position& state, Action action,
    double reward, const Position& nextState) {
    // Current Q-value
    double currentQ = getQ(state, action);

    // Max Q-value for next state
    double maximumNextQ = getQ(nextState, UP);
    for (int a = 1; a < 4; a++) {
        maximumNextQ = std::max(maximumNextQ, getQ(nextState, static_cast<Action>(a)));
    }

    // Bellman equation: newQValue = currentQValue + alpha(reward + gamma * maxNextQValue - currentQValue)
    double newQ = currentQ + alpha * (reward + gamma * maximumNextQ - currentQ);

	// Update Q-table with new Q-value for this state-action pair
    qTable[std::make_pair(state, action)] = newQ;
}

// Epsilon needs to decay over time to allow the agent to shift from exploration to exploitation as it learns
// This function reduces epsilon by multiplying it with a decay rate, but ensures it doesn't go below a minimum threshold
void QLearningAgent::decayEpsilon(double decayRate, double minimumEpsilon) {
    epsilon = std::max(minimumEpsilon, epsilon * decayRate);
}

void QLearningAgent::printPolicy(int gridSize) const {
    std::cout << "\nLearned Policy (best action at each position):\n";

    for (int y = 0; y < gridSize; y++) {
        for (int x = 0; x < gridSize; x++) {
            Position pos{ x, y };
            Action bestAction = UP;
            double bestQ = getQ(pos, UP);

            // Find best action for this position
            for (int a = 1; a < 4; a++) {
                Action action = static_cast<Action>(a);
                if (getQ(pos, action) > bestQ) {
                    bestQ = getQ(pos, action);
                    bestAction = action;
                }
            }

            // Display as arrow
            char arrow;
            switch (bestAction) {
            case UP:    arrow = '^'; break;
            case RIGHT: arrow = '>'; break;
            case DOWN:  arrow = 'v'; break;
            case LEFT:  arrow = '<'; break;
            }

            // Mark goal position
            if (x == gridSize - 1 && y == gridSize - 1) {
                std::cout << "G ";
            }
            else {
                std::cout << arrow << " ";
            }
        }
        std::cout << "\n";
    }
}

void QLearningAgent::printQValues(int gridSize) const {
    std::cout << "\nQ-Values at each position (showing max Q):\n";

    for (int y = 0; y < gridSize; y++) {
        for (int x = 0; x < gridSize; x++) {
            Position pos{ x, y };
            double maxQ = getQ(pos, UP);

            // Find max Q-value over all actions
            for (int a = 1; a < 4; a++) {
                maxQ = std::max(maxQ, getQ(pos, static_cast<Action>(a)));
            }

			// std:: fixed: always show (n) digits after decimal point
			// std:: setprecision(2): show 2 digits after decimal point
			// std:: setw(6): set width of output to 6 characters (for alignment in grid on console)
            std::cout << std::fixed << std::setprecision(2)
                << std::setw(6) << maxQ << " ";
        }
        std::cout << "\n";
    }
}