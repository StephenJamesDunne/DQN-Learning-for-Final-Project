# DQN-Learning-for-Final-Project

## Phase 1: Learning Q-Tables and applying them to Gridworld cpp game.
First steps here are to implement basic q-value tabulating and applying the values therein to a simple DQN Agent.

## Learning Outcomes for Phase 1:
1. Basics of Q-Learning: Q-table, a lookup table mapping (state, action) to expected reward for taking said action.
2. The Bellman equation, and how it updates Q-values.

### Q-values/Q(state, action) = expected sum of all future rewards after taking the given action.

## Hyperparameters needed for Bellman equation:
- Alpha, controls how much to update Q-values based on new experiences. 
- Gamma, determines how much the agent values future rewards compare to immediate ones:
    - High gamma value (0.99) = agent is more forward-thinking and plans ahead.
    - Low gamma (0.1) = agent is myopic and only cares about immediate rewards.
- Epsilon, the rate at which the agent will explore random actions instead of exploiting rewarding actions.
    - Epsilon greedy policy w/ epsilon decay means the agent will start off taking more exploratory actions and gradually decay to exploiting more often.

# Data Flow
```mermaid
flowchart TD
    Start([Program Start]) --> Main[main.cpp]
    Main --> TrainCall[Call Trainer::train_qlearning<br/>episodes=2000, verbose=true]
    
    TrainCall --> CreateEnv[Create GridWorld env<br/>8x8 grid, goal at 7,7]
    TrainCall --> CreateAgent[Create QLearningAgent<br/>α=0.1, γ=0.99, ε=0.5]
    
    CreateEnv --> TrainLoop{Episode Loop<br/>0 to 2000}
    CreateAgent --> TrainLoop
    
    TrainLoop -->|Each Episode| Reset[env.reset<br/>Start at 0,0]
    Reset --> StepLoop{Step Loop<br/>Max 200 steps}
    
    StepLoop -->|Each Step| ChooseAction[agent.choose_action<br/>ε-greedy policy]
    ChooseAction --> EnvStep[env.step<br/>Execute action]
    EnvStep --> Update[agent.update<br/>Bellman equation]
    Update --> CheckDone{Reached<br/>Goal?}
    
    CheckDone -->|No| NextStep[state = next_state]
    NextStep --> StepLoop
    CheckDone -->|Yes| DecayEps[agent.decay_epsilon<br/>ε *= 0.995]
    
    DecayEps --> Print{Episode % 100<br/>== 0?}
    Print -->|Yes| PrintProgress[Print progress stats]
    Print -->|No| NextEpisode
    PrintProgress --> NextEpisode[Next Episode]
    NextEpisode --> TrainLoop
    
    TrainLoop -->|Done| ReturnAgent[Return trained agent<br/>with filled Q-table]
    ReturnAgent --> TestCall[Call Trainer::test_policy]
    TestCall --> TestLoop{Test Episodes<br/>ε=0.0}
    TestLoop --> ShowResults[Display test results]
    ShowResults --> End([Program End])
    
    style Main fill:#e1f5ff
    style TrainLoop fill:#fff4e1
    style StepLoop fill:#ffe1e1
    style Update fill:#e1ffe1
    style ReturnAgent fill:#f0e1ff
```

## Bellman Equation as it relates to Gridworld:

### newQValue = currentQValue + alpha(reward + gamma * maxNextQValue - currentQValue)

## Learning output after 2000 episodes of traning Q-Learning Agent to play Gridworld:
![Output of Gridworld Q-Learning Agent](image.png)

## Best actions policy applied to 8x8 grid (agent always starts at 0, 0):
![Output of completed training for Gridworld Q-Learning Agent](image-1.png)

## Applying tabulated Q-learning outcomes after 2000 episodes to a fresh agent. Fresh agent always takes shortest 14 steps to goal!
![Output of new agent playing Gridworld applying learned knowledge](image-2.png)