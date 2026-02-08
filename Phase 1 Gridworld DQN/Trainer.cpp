#include "Trainer.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

QLearningAgent Trainer::trainQLearning(int num_episodes, bool verbose) {
    GridWorld env(8);
    QLearningAgent agent(0.1, 0.99, 0.5);

    if (verbose) {
        std::cout << "Training Q-Learning agent on 8x8 GridWorld...\n";
        std::cout << "Goal: Reach position (7,7) from (0,0)\n";
        std::cout << "Hyperparameters:\n";
        std::cout << "  - Learning rate (alpha): 0.1\n";
        std::cout << "  - Discount factor (gamma): 0.99\n";
        std::cout << "  - Initial exploration (epsilon): 0.5\n\n";
    }

	// Track rewards for progress reporting
    std::vector<double> episode_rewards;

	// Loop over episodes
    for (int episode = 0; episode < num_episodes; episode++) {

		// Reset environment and initialize variables for this episode
        Position state = env.reset();
		double total_reward = 0.0; // Total reward accumulated in this episode
		int steps = 0;  // Count steps taken in this episode

        // Run episode
        for (int step = 0; step < 200; step++) {
            Action action = agent.chooseAction(state);
            auto [next_state, reward, done] = env.step(action);

            agent.update(state, action, reward, next_state);

            state = next_state;
            total_reward += reward;
            steps++;

            if (done) break;
        }

        episode_rewards.push_back(total_reward);
        agent.decayEpsilon(0.995, 0.01);

        // Print progress every 100 episodes
        if (verbose && episode % 100 == 0) {
            printProgress(episode, episode_rewards, steps, agent.getEpsilon());
        }
    }

    if (verbose) {
        std::cout << "\n=== Training Complete ===\n";
        agent.printPolicy(env.get_grid_size());
        agent.printQValues(env.get_grid_size());
    }

    return agent;
}

void Trainer::testPolicy(QLearningAgent& agent, GridWorld& env, int num_episodes) {
    std::cout << "\n\nTesting learned policy (greedy, no exploration):\n";

    // Temporarily disable exploration
    double original_epsilon = agent.getEpsilon();
    agent.setEpsilon(0.0);

    for (int episode = 0; episode < num_episodes; episode++) {
        Position state = env.reset();
        std::cout << "Episode " << episode << ": ";
        std::cout << "(" << state.x << "," << state.y << ")";

        for (int step = 0; step < 20; step++) {
            Action action = agent.chooseAction(state);
            auto [next_state, reward, done] = env.step(action);
            state = next_state;

            std::cout << " -> (" << state.x << "," << state.y << ")";

            if (done) {
                std::cout << " GOAL!\n";
                break;
            }
        }

        if (state != env.get_goal()) {
            std::cout << " (did not reach goal)\n";
        }
    }

    // Restore original epsilon
    agent.setEpsilon(original_epsilon);
}

void Trainer::printProgress(int episode,
    const std::vector<double>& episode_rewards,
    int steps,
    double epsilon) {
    // Calculate average reward over last 100 episodes
    int start_idx = std::max(0, episode - 99);
    double avg_reward = 0.0;

    for (int i = start_idx; i <= episode; i++) {
        avg_reward += episode_rewards[i];
    }
    avg_reward /= (episode - start_idx + 1);

    std::cout << "Episode " << std::setw(4) << episode
        << " | Avg Reward: " << std::fixed << std::setprecision(3) << avg_reward
        << " | Steps: " << std::setw(3) << steps
        << " | Epsilon: " << std::setprecision(3) << epsilon
        << "\n";
}