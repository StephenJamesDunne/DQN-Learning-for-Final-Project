#include "ReplayBuffer.h"

ReplayBuffer::ReplayBuffer(size_t maxSize)
    : capacity(maxSize), position(0) {
    std::random_device rd;
    rng = std::mt19937(rd());
    buffer.reserve(maxSize);
}

// If the buffer isn't full, add new experience to the end.
// If it is full, overwrite the oldest experience.
// Oldest experience in large buffers ideally should end up being less relevant,
// so this is a simple way to keep the buffer fresh without needing to shift all elements.
void ReplayBuffer::add(const Experience& exp) {
    if (buffer.size() < capacity) {
        buffer.push_back(exp);
    }
    else {
        // Circular buffer: overwrite oldest
        buffer[position] = exp;
    }
    position = (position + 1) % capacity;
}

// Sample a random batch of experiences for training
// Uses random sampling without replacement to ensure diversity in training data
// If there are not enough experiences in the buffer, returns an empty batch
const std::vector<Experience>& ReplayBuffer::sample(int batchSize) {
    sampledBatch.clear();
    sampledBatch.reserve(batchSize);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, size() - 1);

    for (int i = 0; i < batchSize; i++) {
        int idx = dis(gen);
        sampledBatch.push_back(buffer[idx]);
    }

    return sampledBatch;
}