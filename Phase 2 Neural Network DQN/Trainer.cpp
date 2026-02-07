#include "Trainer.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

// Q-Learning Training Phase (From Phase 1)
QLearningAgent Trainer::train_qlearning(int num_episodes, bool verbose) {
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
        agent.decay_epsilon(0.995, 0.01);

        // Print progress every 100 episodes
        if (verbose && episode % 100 == 0) {
            print_progress(episode, episode_rewards, steps, agent.get_epsilon());
        }
    }

    if (verbose) {
        std::cout << "\n=== Q-Learning Training Complete ===\n";
        agent.print_policy(env.get_grid_size());
        agent.print_q_values(env.get_grid_size());
    }

    return agent;
}

void Trainer::test_policy(QLearningAgent& agent, GridWorld& env, int num_episodes) {
    std::cout << "\n\nTesting Q-Learning policy (greedy, no exploration):\n";

    double original_epsilon = agent.get_epsilon();
    agent.set_epsilon(0.0);

    int successful = 0;
    int total_steps = 0;

    for (int episode = 0; episode < num_episodes; episode++) {
        Position state = env.reset();
        std::cout << "Episode " << episode << ": ";
        std::cout << "(" << state.x << "," << state.y << ")";

        int steps = 0;
        for (int step = 0; step < 20; step++) {
            Action action = agent.choose_action(state);
            auto [next_state, reward, done] = env.step(action);
            state = next_state;
            steps++;

            std::cout << " -> (" << state.x << "," << state.y << ")";

            if (done) {
                std::cout << " GOAL! (" << steps << " steps)\n";
                successful++;
                total_steps += steps;
                break;
            }
        }

        if (state != env.get_goal()) {
            std::cout << " (failed)\n";
        }
    }

    std::cout << "\nSuccess rate: " << successful << "/" << num_episodes;
    if (successful > 0) {
        std::cout << " | Avg steps: " << (total_steps / successful);
    }
    std::cout << "\n";

    agent.set_epsilon(original_epsilon);
}

// DQN Training Phase (From Phase 2)

DQNAgent Trainer::train_dqn(int num_episodes, bool verbose) {
    GridWorld env(8);
    DQNAgent agent(0.001, 0.99, 1.0);  // Higher initial epsilon for DQN

    if (verbose) {
        std::cout << "Training DQN agent on 8x8 GridWorld...\n";
        std::cout << "Goal: Reach position (7,7) from (0,0)\n";
        std::cout << "Network: 4 -> 16 -> 16 -> 4\n";
        std::cout << "Hyperparameters:\n";
        std::cout << "  - Learning rate: 0.001\n";
        std::cout << "  - Discount factor (gamma): 0.99\n";
        std::cout << "  - Initial exploration (epsilon): 1.0\n";
        std::cout << "  - Batch size: 32\n";
        std::cout << "  - Replay buffer: 10000\n";
        std::cout << "  - Target update frequency: 100 steps\n\n";
    }

    std::vector<double> episode_rewards;

    for (int episode = 0; episode < num_episodes; episode++) {
        Position state = env.reset();
        Position goal = env.get_goal();
        double total_reward = 0.0;
        int steps = 0;

        // Run episode
        for (int step = 0; step < 200; step++) {
            // Choose action
            Action action = agent.chooseAction(state, goal);

            // Take action in environment
            auto [next_state, reward, done] = env.step(action);

            // Store experience in replay buffer
            agent.remember(state, action, reward, next_state, done, goal);

            // Train on batch from replay buffer
            agent.trainStep();

			// Increment step count (updates target network every 100 steps)
            agent.incrementStep();

            state = next_state;
            total_reward += reward;
            steps++;

            if (done) break;
        }

        episode_rewards.push_back(total_reward);
        agent.decayEpsilon();

        // Print progress every 100 episodes
        if (verbose && episode % 100 == 0) {
            print_dqn_progress(episode, episode_rewards, steps, agent.getEpsilon(), agent);
        }
    }

    if (verbose) {
        std::cout << "\n=== DQN Training Complete ===\n";
    }

    return agent;
}

void Trainer::test_dqn_policy(DQNAgent& agent, GridWorld& env, int num_episodes) {
    std::cout << "\n\nTesting DQN policy (greedy, no exploration):\n";

    double original_epsilon = agent.getEpsilon();
    agent.setEpsilon(0.0);

    int successful = 0;
    int total_steps = 0;
    Position goal = env.get_goal();

    for (int episode = 0; episode < num_episodes; episode++) {
        Position state = env.reset();
        std::cout << "Episode " << episode << ": ";
        std::cout << "(" << state.x << "," << state.y << ")";

        int steps = 0;
        for (int step = 0; step < 20; step++) {
            Action action = agent.chooseAction(state, goal);
            auto [next_state, reward, done] = env.step(action);
            state = next_state;
            steps++;

            std::cout << " -> (" << state.x << "," << state.y << ")";

            if (done) {
                std::cout << " GOAL! (" << steps << " steps)\n";
                successful++;
                total_steps += steps;
                break;
            }
        }

        if (state != env.get_goal()) {
            std::cout << " (failed)\n";
        }
    }

    std::cout << "\nSuccess rate: " << successful << "/" << num_episodes;
    if (successful > 0) {
        std::cout << " | Avg steps: " << (total_steps / successful);
    }
    std::cout << "\n";

    agent.setEpsilon(original_epsilon);
}

// Helper functions

void Trainer::print_progress(int episode,
    const std::vector<double>& episode_rewards,
    int steps,
    double epsilon) {
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

void Trainer::print_dqn_progress(int episode,
    const std::vector<double>& episode_rewards,
    int steps,
    double epsilon,
    const DQNAgent& agent) {
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
        << " | Buffer: " << agent.getBufferSize() << " experiences\n";
}