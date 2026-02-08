#include "Trainer.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

// Q-Learning Training Phase (From Phase 1)
QLearningAgent Trainer::trainQLearning(int numEpisodes, bool verbose) {
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

    std::vector<double> episodeRewards;

    for (int episode = 0; episode < numEpisodes; episode++) {
        Position state = env.reset();
        double totalReward = 0.0;
        int steps = 0;

        // Run episode
        for (int step = 0; step < 200; step++) {
            Action action = agent.chooseAction(state);
            auto [nextState, reward, done] = env.step(action);

            agent.update(state, action, reward, nextState);

            state = nextState;
            totalReward += reward;
            steps++;

            if (done) break;
        }

        episodeRewards.push_back(totalReward);
        agent.decayEpsilon(0.995, 0.01);

        // Print progress every 100 episodes
        if (verbose && episode % 100 == 0) {
            printProgress(episode, episodeRewards, steps, agent.getEpsilon());
        }
    }

    if (verbose) {
        std::cout << "\n=== Q-Learning Training Complete ===\n";
        agent.printPolicy(env.get_grid_size());
        agent.printQValues(env.get_grid_size());
    }

    return agent;
}

void Trainer::testPolicy(QLearningAgent& agent, GridWorld& env, int num_episodes) {
    std::cout << "\n\nTesting Q-Learning policy (greedy, no exploration):\n";

    double originalEpsilon = agent.getEpsilon();
    agent.setEpsilon(0.0);

    int successful = 0;
    int totalSteps = 0;

    for (int episode = 0; episode < num_episodes; episode++) {
        Position state = env.reset();
        std::cout << "Episode " << episode << ": ";
        std::cout << "(" << state.x << "," << state.y << ")";

        int steps = 0;
        for (int step = 0; step < 20; step++) {
            Action action = agent.chooseAction(state);
            auto [nextState, reward, done] = env.step(action);
            state = nextState;
            steps++;

            std::cout << " -> (" << state.x << "," << state.y << ")";

            if (done) {
                std::cout << " GOAL! (" << steps << " steps)\n";
                successful++;
                totalSteps += steps;
                break;
            }
        }

        if (state != env.get_goal()) {
            std::cout << " (failed)\n";
        }
    }

    std::cout << "\nSuccess rate: " << successful << "/" << num_episodes;
    if (successful > 0) {
        std::cout << " | Avg steps: " << (totalSteps / successful);
    }
    std::cout << "\n";

    agent.setEpsilon(originalEpsilon);
}

// DQN Training Phase (From Phase 2)

DQNAgent Trainer::trainDQN(int num_episodes, bool verbose) {
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

    std::vector<double> episodeRewards;

    for (int episode = 0; episode < num_episodes; episode++) {
        Position state = env.reset();
        Position goal = env.get_goal();
        double totalReward = 0.0;
        int steps = 0;

        // Run episode
        for (int step = 0; step < 200; step++) {
            // Choose action
            Action action = agent.chooseAction(state, goal);

            // Take action in environment
            auto [nextState, reward, done] = env.step(action);

            // Store experience in replay buffer
            agent.remember(state, action, reward, nextState, done, goal);

            // Train on batch from replay buffer
            agent.trainStep();

			// Increment step count (updates target network every 100 steps)
            agent.incrementStep();

            state = nextState;
            totalReward += reward;
            steps++;

            if (done) break;
        }

        episodeRewards.push_back(totalReward);
        agent.decayEpsilon();

        // Print progress every 100 episodes
        if (verbose && episode % 100 == 0) {
            printDQNProgress(episode, episodeRewards, steps, agent.getEpsilon(), agent);
        }
    }

    if (verbose) {
        std::cout << "\n=== DQN Training Complete ===\n";
    }

    return agent;
}

void Trainer::testDQNPolicy(DQNAgent& agent, GridWorld& env, int num_episodes) {
    std::cout << "\n\nTesting DQN policy (greedy, no exploration):\n";

    double originalEpsilon = agent.getEpsilon();
    agent.setEpsilon(0.0);

    int successful = 0;
    int totalSteps = 0;
    Position goal = env.get_goal();

    for (int episode = 0; episode < num_episodes; episode++) {
        Position state = env.reset();
        std::cout << "Episode " << episode << ": ";
        std::cout << "(" << state.x << "," << state.y << ")";

        int steps = 0;
        for (int step = 0; step < 20; step++) {
            Action action = agent.chooseAction(state, goal);
            auto [nextState, reward, done] = env.step(action);
            state = nextState;
            steps++;

            std::cout << " -> (" << state.x << "," << state.y << ")";

            if (done) {
                std::cout << " GOAL! (" << steps << " steps)\n";
                successful++;
                totalSteps += steps;
                break;
            }
        }

        if (state != env.get_goal()) {
            std::cout << " (failed)\n";
        }
    }

    std::cout << "\nSuccess rate: " << successful << "/" << num_episodes;
    if (successful > 0) {
        std::cout << " | Avg steps: " << (totalSteps / successful);
    }
    std::cout << "\n";

    agent.setEpsilon(originalEpsilon);
}

// Helper functions

void Trainer::printProgress(int episode,
    const std::vector<double>& episodeRewards,
    int steps,
    double epsilon) {
    int startIndex = std::max(0, episode - 99);
    double averageReward = 0.0;

    for (int i = startIndex; i <= episode; i++) {
        averageReward += episodeRewards[i];
    }
    averageReward /= (episode - startIndex + 1);

    std::cout << "Episode " << std::setw(4) << episode
        << " | Avg Reward: " << std::fixed << std::setprecision(3) << averageReward
        << " | Steps: " << std::setw(3) << steps
        << " | Epsilon: " << std::setprecision(3) << epsilon
        << "\n";
}

void Trainer::printDQNProgress(int episode,
    const std::vector<double>& episodeRewards,
    int steps,
    double epsilon,
    const DQNAgent& agent) {
    int startIndex = std::max(0, episode - 99);
    double averageReward = 0.0;

    for (int i = startIndex; i <= episode; i++) {
        averageReward += episodeRewards[i];
    }
    averageReward /= (episode - startIndex + 1);

    std::cout << "Episode " << std::setw(4) << episode
        << " | Avg Reward: " << std::fixed << std::setprecision(3) << averageReward
        << " | Steps: " << std::setw(3) << steps
        << " | Epsilon: " << std::setprecision(3) << epsilon
        << " | Buffer: " << agent.getBufferSize() << " experiences\n";
}