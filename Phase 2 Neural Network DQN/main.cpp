#include "GridWorld.h"
#include "QLearningAgent.h"
#include "DQNAgent.h"
#include "Trainer.h"
#include <iostream>

void print_menu() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Gridworld RL Training System\n";
    std::cout << "========================================\n\n";
    std::cout << "Choose what to run:\n\n";
    std::cout << "  1. Q-Learning (Phase 1) - Tabular method\n";
    std::cout << "  2. DQN (Phase 2) - Neural network method\n";
    std::cout << "  3. Both (compare side-by-side)\n";
    std::cout << "  4. Exit\n\n";
    std::cout << "Enter choice (1-4): ";
}

void run_qlearning() {
    std::cout << "\n========================================\n";
    std::cout << "  Q-LEARNING (Tabular Method)\n";
    std::cout << "========================================\n\n";

    std::cout << "How it works:\n";
    std::cout << "  - Stores exact Q-values for each (state, action) pair\n";
    std::cout << "  - Updates using Bellman equation: Q(s,a) += alpha * [r + gamma*max(Q(s')) - Q(s,a)]\n";
    std::cout << "  - Fast to train but doesn't generalize to new states\n\n";

    QLearningAgent agent = Trainer::trainQLearning(2000, true);

    GridWorld test_env(8);
    Trainer::testPolicy(agent, test_env, 10);

    std::cout << "\n\nPress Enter to continue...";
    std::cin.ignore();
    std::cin.get();
}

void run_dqn() {
    std::cout << "\n========================================\n";
    std::cout << "  DQN (Neural Network Method)\n";
    std::cout << "========================================\n\n";

    std::cout << "How it works:\n";
    std::cout << "  - Neural network learns to predict Q-values from state features\n";
    std::cout << "  - Network structure: 4 inputs -> 16 hidden -> 16 hidden -> 4 outputs\n";
    std::cout << "  - Uses experience replay (stores past experiences, learns from random batches)\n";
    std::cout << "  - Uses target network (stable learning target, updated every 100 steps)\n";
    std::cout << "  - Can generalize to similar states, but needs more training data\n\n";

    std::cout << "What to watch for:\n";
    std::cout << "  - Avg Reward: Starts at -2.0 (max steps penalty), should rise to ~0.85\n";
    std::cout << "  - Steps: Starts at 200 (max), should drop to ~14-15 (optimal path)\n";
    std::cout << "  - Epsilon: Starts at 1.0 (100% random), decays to 0.01 (1% random)\n";
    std::cout << "  - Learning is slower and noisier than Q-Learning due to:\n";
    std::cout << "      * Neural network requires many gradient updates\n";
    std::cout << "      * Random batch sampling adds variance\n";
    std::cout << "      * Network must learn feature representations\n\n";

    std::cout << "Press Enter to start training...";
    std::cin.ignore();
    std::cin.get();

    DQNAgent agent = Trainer::trainDQN(1000, true);

    GridWorld test_env(8);
    Trainer::testDQNPolicy(agent, test_env, 10);

    std::cout << "\n\nPress Enter to continue...";
    std::cin.get();
}

void run_both() {
    std::cout << "\n========================================\n";
    std::cout << "  COMPARISON: Q-Learning vs DQN\n";
    std::cout << "========================================\n\n";

    // Q-Learning
    std::cout << "--- PHASE 1: Q-Learning ---\n";
    QLearningAgent q_agent = Trainer::trainQLearning(2000, true);
    GridWorld test_env_q(8);
    Trainer::testPolicy(q_agent, test_env_q, 10);

    std::cout << "\n\n";

    // DQN
    std::cout << "--- PHASE 2: DQN ---\n";
    DQNAgent dqn_agent = Trainer::trainDQN(5000, true);
    GridWorld test_env_dqn(8);
    Trainer::testDQNPolicy(dqn_agent, test_env_dqn, 10);

    std::cout << "\n\n========================================\n";
    std::cout << "  Key Differences Summary\n";
    std::cout << "========================================\n\n";
    std::cout << "Q-Learning:\n";
    std::cout << "  + Faster training (2000 episodes)\n";
    std::cout << "  + More stable learning curve\n";
    std::cout << "  + Perfect for small, discrete state spaces\n";
    std::cout << "  - Cannot generalize to new states\n";
    std::cout << "  - Scales poorly (memory grows with state space)\n\n";

    std::cout << "DQN:\n";
    std::cout << "  + Can generalize to similar/unseen states\n";
    std::cout << "  + Fixed memory (network size doesn't grow)\n";
    std::cout << "  + Scales to large/continuous state spaces\n";
    std::cout << "  - Slower training (needs 5000+ episodes)\n";
    std::cout << "  - More hyperparameters to tune\n";
    std::cout << "  - Learning curve is noisier\n\n";

    std::cout << "Press Enter to continue...";
    std::cin.get();
}

int main() {
    while (true) {
        print_menu();

        int choice;
        std::cin >> choice;

        switch (choice) {
        case 1:
            run_qlearning();
            break;
        case 2:
            run_dqn();
            break;
        case 3:
            run_both();
            break;
        case 4:
            std::cout << "\nExiting...\n";
            return 0;
        default:
            std::cout << "\nInvalid choice. Please enter 1-4.\n";
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            break;
        }
    }

    return 0;
}