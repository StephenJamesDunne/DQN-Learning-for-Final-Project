#pragma once

// 2D position in the gridworld
struct Position {
    int x, y;

    // Operators used for map keys and comparisons
    bool operator==(const Position& other) const {
        return x == other.x && y == other.y;
    }

    bool operator!=(const Position& other) const {
        return !(*this == other);
    }

    bool operator<(const Position& other) const {
        return (x < other.x) || (x == other.x && y < other.y);
    }
};

// Actions available in the gridworld
enum Action {
    UP = 0,
    RIGHT = 1,
    DOWN = 2,
    LEFT = 3
};