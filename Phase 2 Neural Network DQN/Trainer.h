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
    static QLearningAgent trainQLearning(int numEpisodes = 2000,
        bool verbose = true);

    /**
     * Train DQN agent with neural network
     */
    static DQNAgent trainDQN(int numEpisodes = 5000,
        bool verbose = true);

    /**
     * Test Q-Learning agent's policy
     */
    static void testPolicy(QLearningAgent& agent,
        GridWorld& env,
        int numEpisodes = 10);

    /**
     * Test DQN agent's policy
     */
    static void testDQNPolicy(DQNAgent& agent,
        GridWorld& env,
        int numEpisodes = 10);

private:
    /**
     * Print training progress for Q-Learning
     */
    static void printProgress(int episode,
        const std::vector<double>& episodeRewards,
        int steps,
        double epsilon);

    /**
     * Print training progress for DQN
     */
    static void printDQNProgress(int episode,
        const std::vector<double>& episodeRewards,
        int steps,
        double epsilon,
        const DQNAgent& agent);
};