#include "GridWorld.h"
#include <algorithm>

GridWorld::GridWorld(int size)
    : grid_size(size), goal{ size - 1, size - 1 } {
    std::random_device rd;
    rng = std::mt19937(rd());
    reset();
}

Position GridWorld::reset() {
    current_state = { 0, 0 };
    return current_state;
}

// Takes an action and returns the new state, reward, and whether the episode is done
std::tuple<Position, double, bool> GridWorld::step(Action action) {
    // Copy the current state to modify it based on the action
    Position next_state = current_state;

    // Apply action with boundary checking
    switch (action) {
    case UP:
        next_state.y = std::max(0, current_state.y - 1);
        break;
    case RIGHT:
        next_state.x = std::min(grid_size - 1, current_state.x + 1);
        break;
    case DOWN:
        next_state.y = std::min(grid_size - 1, current_state.y + 1);
        break;
    case LEFT:
        next_state.x = std::max(0, current_state.x - 1);
        break;
    }

    // Update current state
    current_state = next_state;

    // Reward: +1 for reaching the goal, -0.01 for each step to encourage shorter paths
    double reward = (current_state == goal) ? 1.0 : -0.01;
    bool done = (current_state == goal);

    return { current_state, reward, done };
}