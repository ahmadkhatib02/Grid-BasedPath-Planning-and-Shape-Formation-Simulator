# MARL Enhancements for Shape Formation

This document explains the enhancements made to the Multi-Agent Reinforcement Learning (MARL) system for improved shape formation with a special focus on path blocking scenarios.

## Key Enhancements

### 1. Path Blocking Detection and Handling

New mechanisms specifically designed to handle path blocking:
- **Path blocking detection**: Agents can detect when they are blocking another agent's path
- **Path clearing rewards**: Agents are rewarded for moving out of the way of other agents
- **Waiting behavior**: Agents learn to wait when their path is blocked instead of thrashing
- **Corridor detection**: Special handling for tight spaces where agents must coordinate

### 2. Enhanced Reward Structure

The reward function has been optimized for parallel movement with path blocking:
- **Cooperative bonuses**: Agents receive additional rewards when multiple agents reach their targets
- **Formation rewards**: Agents are rewarded for maintaining appropriate distances from each other
- **Global formation improvement**: Rewards for actions that reduce the overall formation error
- **Path clearing incentives**: Specific rewards for moving out of other agents' paths

### 3. Advanced State Representation

Agents now have a much richer view of their environment:
- **Path information**: Agents can "see" the direct path to their target
- **Blockage detection**: Agents can detect when their path is blocked by other agents
- **Other agents' targets**: Agents can "see" where other agents are trying to go
- **Relative positions**: Detailed information about nearest agents and their movement intentions

### 4. Specialized Neural Network Architecture

The DQN architecture has been significantly enhanced:
- **Dueling DQN**: Separate advantage and value streams for better policy learning
- **Deeper network**: More layers for better feature extraction
- **Specialized path processing**: Dedicated neural network branches for processing path information
- **Huber loss**: More robust loss function for handling outliers in the training data

### 5. Specialized Training Scenarios

Three types of challenging scenarios for training path blocking behavior:
- **Corridor scenarios**: Agents must navigate through narrow passages
- **Crossing paths scenarios**: Agents must cross each other's paths efficiently
- **Bottleneck scenarios**: Multiple agents must navigate through a tight space

### 6. Curriculum Learning and Staged Training

Progressive learning approaches for better skill acquisition:
- **Curriculum learning**: Gradually increasing difficulty as training progresses
- **Staged training**: Focus on different skills at different stages of training
- **Adaptive difficulty**: Difficulty adjusts based on agent performance

### 7. Optimized Hyperparameters

Fine-tuned parameters for path blocking scenarios:
- **Higher discount factor**: Better long-term planning (0.99)
- **Lower learning rate**: More stable learning in complex scenarios
- **Larger memory size**: Better experience diversity for handling rare situations

## How to Use

1. **Basic Training**: Click "Train MARL for Parallel Movement" for general training
   - Recommended for initial learning of basic movement patterns
   - 300+ episodes recommended

2. **Path Blocking Training**: Click "Train for Path Blocking" for specialized training
   - Use this after basic training to improve path blocking behavior
   - Creates challenging scenarios that focus on agent coordination
   - Automatically cycles through corridor, crossing paths, and bottleneck scenarios

3. **Using Trained Agents**: After training, click "Do Shape with MARL" to use the trained agents

4. **Monitoring Progress**: The training progress is displayed in real-time, and detailed metrics are shown in the console

## Tips for Best Results

- **Combined training approach**: Start with basic training, then use path blocking training
- **Use more episodes**: 300-500 episodes are recommended for complex shapes
- **Keep curriculum learning enabled**: This significantly improves learning efficiency
- **Add some obstacles**: The agents learn better when there are obstacles to navigate around
- **Start with simple shapes**: Train on simpler shapes before attempting complex ones
- **Create bottlenecks**: Add obstacles that create narrow passages to test coordination
