#pragma once
#include "GridWorld.h"
#include "QLearningAgent.h"
#include "DQNAgent.h"

/**
 * Training utilities for both Q-Learning and DQN agents
 */
class Trainer {
public:
    /**
     * Train Q-Learning agent (from Phase 1)
     */
    static QLearningAgent train_qlearning(int num_episodes = 2000,
        bool verbose = true);

    /**
     * Train DQN agent with neural network
     */
    static DQNAgent train_dqn(int num_episodes = 5000,
        bool verbose = true);

    /**
     * Test Q-Learning agent's policy
     */
    static void test_policy(QLearningAgent& agent,
        GridWorld& env,
        int num_episodes = 10);

    /**
     * Test DQN agent's policy
     */
    static void test_dqn_policy(DQNAgent& agent,
        GridWorld& env,
        int num_episodes = 10);

private:
    /**
     * Print training progress for Q-Learning
     */
    static void print_progress(int episode,
        const std::vector<double>& episode_rewards,
        int steps,
        double epsilon);

    /**
     * Print training progress for DQN
     */
    static void print_dqn_progress(int episode,
        const std::vector<double>& episode_rewards,
        int steps,
        double epsilon,
        const DQNAgent& agent);
};