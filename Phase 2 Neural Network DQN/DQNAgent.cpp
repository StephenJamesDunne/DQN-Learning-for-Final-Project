#include "DQNAgent.h"
#include <algorithm>
#include <iostream>

DQNAgent::DQNAgent(double lr, double discount, double exploration)
    : qNetwork({ 4, 16, 16, 4 }),
    targetNetwork({ 4, 16, 16, 4 }),
    replayBuffer(10000),
    learningRate(lr),
    gamma(discount),
    epsilon(exploration),
    epsilonDecay(0.995),
    epsilonMinimum(0.01),
    batchSize(32),
    targetUpdateFrequency(100),
    stepsSinceTargetUpdate(0),
	totalSteps(0),
    uniformDist(0.0, 1.0)
{
    std::random_device rd;
    rng = std::mt19937(rd());

    // Initialize target network with same weights as Q-network
    targetNetwork.copyWeightsFrom(qNetwork);
}

std::vector<double> DQNAgent::stateToInput(const Position& state, const Position& goal) const {
    // Normalize positions to [0, 1]
    return {
        state.x / 8.0,
        state.y / 8.0,
        goal.x / 8.0,
        goal.y / 8.0
    };
}

Action DQNAgent::chooseAction(const Position& state, const Position& goal) {
    // Exploration: random action
    if (uniformDist(rng) < epsilon) {
        return static_cast<Action>(rng() % 4);
    }

    // Exploitation: choose best action from Q-network
    std::vector<double> input = stateToInput(state, goal);
    std::vector<double> qValues = qNetwork.forward(input);

    // Find action with max Q-value
    int bestAction = 0;
    double maxQ = qValues[0];

    for (int a = 1; a < 4; a++) {
        if (qValues[a] > maxQ) {
            maxQ = qValues[a];
            bestAction = a;
        }
    }

    return static_cast<Action>(bestAction);
}

void DQNAgent::remember(const Position& state, Action action, double reward,
    const Position& nextState, bool done, const Position& goal) {
    Experience exp;
    exp.state = stateToInput(state, goal);
    exp.action = static_cast<int>(action);
    exp.reward = reward;
    exp.nextState = stateToInput(nextState, goal);
    exp.done = done;

    replayBuffer.add(exp);
}

void DQNAgent::trainStep() {
    // Don't train if not enough experiences
    if (!replayBuffer.canSample(batchSize)) {
        return;
    }

    // Sample random batch from replay buffer
    const std::vector<Experience>& batch = replayBuffer.sample(batchSize);

    // Train on each experience in the batch
    for (const auto& exp : batch) {
        // Compute target Q-value using target network
        std::vector<double> nextQValues = targetNetwork.forward(exp.nextState);
        double maximumNextQ = *std::max_element(nextQValues.begin(), nextQValues.end());

        // Bellman equation
        double targetQValue = exp.done ? exp.reward : exp.reward + gamma * maximumNextQ;

        // Current Q-value from online network
        std::vector<double> currentQValues = qNetwork.forward(exp.state);
        double currentQValue = currentQValues[exp.action];

        // TD error
        double tdError = targetQValue - currentQValue;

        // Backpropagate through Q-network
        qNetwork.backward(exp.state, exp.action, tdError, learningRate);
    }
}

void DQNAgent::incrementStep() {
    totalSteps++;

    if (totalSteps % targetUpdateFrequency == 0) {
        targetNetwork.copyWeightsFrom(qNetwork);
        
        // Only print every 1000 steps to avoid flooding console
        if (totalSteps % 1000 == 0) {
            std::cout << "  [Target network updated at step " << totalSteps << "]\n";
        }
    }
}

void DQNAgent::decayEpsilon() {
    epsilon = std::max(epsilonMinimum, epsilon * epsilonDecay);
}