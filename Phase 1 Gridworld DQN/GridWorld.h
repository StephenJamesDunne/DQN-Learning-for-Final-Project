#pragma once
#include "Position.h"
#include <tuple>
#include <random>

// GridWorld game: Agent starts at (0,0) and must reach goal at (7,7) on an 8x8 grid.
// State: Agent's current position (x,y)
// Actions: UP, RIGHT, DOWN, LEFT
// Rewards: +1 for reaching goal, -0.01 per step to encourage shorter paths
// The GridWorld class manages the environment, including state transitions and rewards.

class GridWorld {
private:
    Position current_state;
    Position goal;
    int grid_size;
    std::mt19937 rng;

public:
	// Constructor with optional grid size (default 8x8)
    explicit GridWorld(int size = 8);

	// Reset the environment to the initial state and return it
    Position reset();

	// Take an action and return the new state, reward, and whether the episode is done
    std::tuple<Position, double, bool> step(Action action);

    Position get_state() const { return current_state; }
    int get_grid_size() const { return grid_size; }
    Position get_goal() const { return goal; }
};