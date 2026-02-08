#pragma once
#include "GridWorld.h"
#include "QLearningAgent.h"

// Training utils for Q-Learning agent on GridWorld game
class Trainer {
public:
    
	// num_episodes: Number of training episodes to run
	// verbose: Whether to print training progress and final policy
    static QLearningAgent trainQLearning(int num_episodes = 10000,
        bool verbose = true);

	// Test the learned policy by running episodes with epsilon=0 (greedy)
	// Greedy = no exploration, always choose best known action
    static void testPolicy(QLearningAgent& agent,
        GridWorld& env,
        int num_episodes = 10);

private:
	// Debugging helper to print training progress every 100 episodes
    static void printProgress(int episode,
        const std::vector<double>& episode_rewards,
        int steps,
        double epsilon);
};