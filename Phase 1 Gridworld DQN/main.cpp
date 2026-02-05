#include <iostream>
#include <map>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>

// 2D position
struct Position {
    int x, y;

	// Operators used for map keys and comparisons
    bool operator==(const Position& other) const {
        return x == other.x && y == other.y;
    }

    bool operator<(const Position& other) const {
        return (x < other.x) || (x == other.x && y < other.y);
    }
};

// Actions in the gridworld
enum Action {
    UP = 0,
    RIGHT = 1,
    DOWN = 2,
    LEFT = 3
};

// GridWorld Environment
class GridWorld {
private:
    Position current_state;
    Position goal;
    int grid_size;
    std::mt19937 rng;

public:
    GridWorld(int size = 8) : grid_size(size), goal{ size - 1, size - 1 } {
        std::random_device rd;
        rng = std::mt19937(rd());
        reset();
    }

    Position reset() {
        current_state = { 0, 0 };
        return current_state;
    }

	// Main environment step function: takes an action and returns (next_state, reward, done)
    std::tuple<Position, double, bool> step(Action action) {

		// Copy the current state to modify it based on the action
        Position next_state = current_state;

        // Apply action
        switch (action) {
        case UP:    next_state.y = std::max(0, current_state.y - 1); break;
        case RIGHT: next_state.x = std::min(grid_size - 1, current_state.x + 1); break;
        case DOWN:  next_state.y = std::min(grid_size - 1, current_state.y + 1); break;
        case LEFT:  next_state.x = std::max(0, current_state.x - 1); break;
        }

		// Apply the action and update the current state
        current_state = next_state;

		// Reward: +1 for reaching the goal, -0.01 for each step to encourage shorter pathss
        double reward = (current_state == goal) ? 1.0 : -0.01;
        bool done = (current_state == goal);

        return { current_state, reward, done };
    }

    Position get_state() const { return current_state; }
    int get_grid_size() const { return grid_size; }
};

// Q-Learning Agent
// Functionality: 
// - Maintains a Q-table mapping (state, action) pairs to Q-values
// - Chooses actions using an epsilon-greedy policy
// - Updates Q-values based on the reward received and the max Q-value of the next state
// - Provides functions to print the learned policy and Q-values for visualization
// Parameters: learning_rate (alpha), discount_factor (gamma), exploration_rate (epsilon)
// Learning rate determines how much the agent updates its Q-values based on new information.
// Discount factor determines how much the agent values future rewards compared to immediate rewards.
// Exploration rate determines how often the agent chooses a random action (exploration) versus the action with the highest Q-value (exploitation).


class QLearningAgent {
private:
    std::map<std::pair<Position, Action>, double> q_table;
    double alpha;      // learning rate
    double gamma;      // discount factor
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_dist;

public:
    double epsilon;    // exploration rate

    QLearningAgent(double learning_rate = 0.1,
        double discount = 0.99,
        double exploration = 0.1)
        : alpha(learning_rate), gamma(discount), epsilon(exploration),
        uniform_dist(0.0, 1.0) {
        std::random_device rd;
        rng = std::mt19937(rd());
    }

    // Get Q-value for a state-action pair
	// find the Q-value for the given state and action. 
    // If it doesn't exist in the Q-table, return a default value (1.0) to encourage exploration.
    double get_q(const Position& state, Action action) {
        auto key = std::make_pair(state, action);
        if (q_table.find(key) == q_table.end()) {
            return 1.0;
        }
        return q_table[key];
    }

    // Choose action using epsilon-greedy policy
    Action choose_action(const Position& state) {
        // Exploration
        if (uniform_dist(rng) < epsilon) {
            return static_cast<Action>(rng() % 4);
        }

        // Exploitation: choose best action
        double best_q = get_q(state, UP);
        Action best_action = UP;

        for (int a = 1; a < 4; a++) {
            Action action = static_cast<Action>(a);
            double q = get_q(state, action);
            if (q > best_q) {
                best_q = q;
                best_action = action;
            }
        }

        return best_action;
    }

    // Update Q-value using Q-learning rule
    void update(const Position& state, Action action,
        double reward, const Position& next_state) {
        // Current Q-value
        double current_q = get_q(state, action);

        // Max Q-value for next state
        double max_next_q = get_q(next_state, UP);
        for (int a = 1; a < 4; a++) {
            max_next_q = std::max(max_next_q, get_q(next_state, static_cast<Action>(a)));
        }

        // Q-learning formula
        // newQValue = currentQValue + alpha(reward + gamma * maxNextQValue - currentQValue)
        double new_q = current_q + alpha * (reward + gamma * max_next_q - current_q);

        q_table[std::make_pair(state, action)] = new_q;
    }

    // Print Q-table for visualization
    void print_policy(int grid_size) {
        std::cout << "\nLearned Policy (best action at each position):\n";
        for (int y = 0; y < grid_size; y++) {
            for (int x = 0; x < grid_size; x++) {
                Position pos{ x, y };
                Action best_action = UP;
                double best_q = get_q(pos, UP);

                for (int a = 1; a < 4; a++) {
                    Action action = static_cast<Action>(a);
                    if (get_q(pos, action) > best_q) {
                        best_q = get_q(pos, action);
                        best_action = action;
                    }
                }

                char arrow;
                switch (best_action) {
                case UP:    arrow = '^'; break;
                case RIGHT: arrow = '>'; break;
                case DOWN:  arrow = 'v'; break;
                case LEFT:  arrow = '<'; break;
                }

                if (x == grid_size - 1 && y == grid_size - 1) {
                    std::cout << "G ";  // Goal
                }
                else {
                    std::cout << arrow << " ";
                }
            }
            std::cout << "\n";
        }
    }

    void print_q_values(int grid_size) {
        std::cout << "\nQ-Values at each position (showing max Q):\n";
        for (int y = 0; y < grid_size; y++) {
            for (int x = 0; x < grid_size; x++) {
                Position pos{ x, y };
                double max_q = get_q(pos, UP);
                for (int a = 1; a < 4; a++) {
                    max_q = std::max(max_q, get_q(pos, static_cast<Action>(a)));
                }
                std::cout << std::fixed << std::setprecision(2)
                    << std::setw(6) << max_q << " ";
            }
            std::cout << "\n";
        }
    }
};

// Training function
void train(int num_episodes = 10000, bool verbose = true) {
    GridWorld env(8);
    QLearningAgent agent(0.1, 0.99, 0.5);

    std::cout << "Training Q-Learning agent on 8x8 GridWorld...\n";
    std::cout << "Goal: Reach position (7,7) from (0,0)\n\n";

    std::vector<double> episode_rewards;

    for (int episode = 0; episode < num_episodes; episode++) {
        Position state = env.reset();
        double total_reward = 0.0;
        int steps = 0;

        // Run episode
        for (int step = 0; step < 200; step++) {
            Action action = agent.choose_action(state);
            auto [next_state, reward, done] = env.step(action);

            agent.update(state, action, reward, next_state);

            state = next_state;
            total_reward += reward;
            steps++;

            if (done) break;
        }

        episode_rewards.push_back(total_reward);

		agent.epsilon = std::max(0.01, agent.epsilon * 0.995);  // Decay exploration rate

        // Print progress
        if (verbose && episode % 100 == 0) {
            // Calculate average reward over last 100 episodes
            int start_idx = std::max(0, episode - 99);
            double avg_reward = 0.0;
            for (int i = start_idx; i <= episode; i++) {
                avg_reward += episode_rewards[i];
            }
            avg_reward /= (episode - start_idx + 1);

            std::cout << "Episode " << std::setw(4) << episode
                << " | Avg Reward: " << std::fixed << std::setprecision(3)
                << avg_reward
                << " | Steps: " << steps
                << " | Epsilon: " << std::setprecision(3) << agent.epsilon
                // =========================
                << "\n";
        }
    }

    // Display learned policy
    agent.print_policy(env.get_grid_size());
    agent.print_q_values(env.get_grid_size());
}

// Test the learned policy
void test_policy(QLearningAgent& agent, GridWorld& env, int num_episodes = 10) {
    std::cout << "\n\nTesting learned policy (no exploration):\n";

    for (int episode = 0; episode < num_episodes; episode++) {
        Position state = env.reset();
        std::cout << "Episode " << episode << ": ";
        std::cout << "(" << state.x << "," << state.y << ")";

        for (int step = 0; step < 20; step++) {
            // Use greedy policy (no exploration)
            Action best_action = UP;
            double best_q = agent.get_q(state, UP);
            for (int a = 1; a < 4; a++) {
                Action action = static_cast<Action>(a);
                if (agent.get_q(state, action) > best_q) {
                    best_q = agent.get_q(state, action);
                    best_action = action;
                }
            }

            auto [next_state, reward, done] = env.step(best_action);
            state = next_state;

            std::cout << " -> (" << state.x << "," << state.y << ")";

            if (done) {
                std::cout << " GOAL!\n";
                break;
            }
        }
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  Q-Learning on GridWorld (C++)\n";
    std::cout << "========================================\n\n";

    train(10000, true);

    // Create a new agent and env for testing
    GridWorld env(8);
	QLearningAgent agent(0.1, 0.99, 0.5);  // epsilon at 0.5 for testing to see some exploration

    // Train it first
    for (int episode = 0; episode < 500; episode++) {
        Position state = env.reset();
        for (int step = 0; step < 100; step++) {
            Action action = agent.choose_action(state);
            auto [next_state, reward, done] = env.step(action);
            agent.update(state, action, reward, next_state);
            state = next_state;
            if (done) break;
        }
    }

    //test_policy(agent, env, 10);

    return std::cin.get();
}