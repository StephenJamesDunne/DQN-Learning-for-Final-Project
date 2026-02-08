#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

// Experience Relay Buffer for DQN Agent
// Stores past experiences (state, action, reward, next_state, done) for training the DQN agent
// Allows the agent to learn from past experiences by sampling random batches for training

struct Experience {
	std::vector<double> state;      // Where the agent was
	int action;                     // What action the agent took
	double reward;                  // What reward the agent received
	std::vector<double> nextState; // Where the agent ended up
	bool done;                      // Whether the episode ended after this experience
};

class ReplayBuffer {
private:
    std::vector<Experience> buffer;
    std::vector<Experience> sampledBatch;
    size_t capacity;
    size_t position;
    std::mt19937 rng;

public:
	// Constructor
    explicit ReplayBuffer(size_t maxSize);

	// Add experience to buffer (overwrites oldest experience if capacity is exceeded)
    void add(const Experience& exp);

	// Sample a random batch of experiences for training
    const std::vector<Experience>& sample(int batchSize);

    size_t size() const { return buffer.size(); }

    bool canSample(size_t batchSize) const {
        return buffer.size() >= batchSize;
    }
};