#include "ReplayBuffer.h"

ReplayBuffer::ReplayBuffer(size_t max_size)
    : capacity(max_size), position(0) {
    std::random_device rd;
    rng = std::mt19937(rd());
    buffer.reserve(max_size);
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
const std::vector<Experience>& ReplayBuffer::sample(int batch_size) {
    sampled_batch.clear();
    sampled_batch.reserve(batch_size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, size() - 1);

    for (int i = 0; i < batch_size; i++) {
        int idx = dis(gen);
        sampled_batch.push_back(buffer[idx]);
    }

    return sampled_batch;
}