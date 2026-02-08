#include "GridWorld.h"
#include "QLearningAgent.h"
#include "Trainer.h"
#include <iostream>

int main() {
    std::cout << "========================================\n";
    std::cout << "  Q-Learning on GridWorld (C++)\n";
    std::cout << "========================================\n\n";

    // Train Q-Learning agent
    QLearningAgent agent = Trainer::trainQLearning(2000, true);

    // Optionally test the learned policy
    GridWorld test_env(8);
    Trainer::testPolicy(agent, test_env, 10);

    std::cout << "\nPress Enter to exit...";
    std::cin.get();
    return 0;
}