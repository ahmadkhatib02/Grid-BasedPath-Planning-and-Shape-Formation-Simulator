# Multi-Agent Reinforcement Learning (MARL) for Path Planning

This module implements Multi-Agent Reinforcement Learning for path planning with obstacles. It can be integrated with your existing grid-based simulation project.

## Files

- `PathPlanningMARL.py` - The core MARL implementation
- `MARL_Integration.py` - A standalone visualization tool for training and testing the MARL system

## Features

- **Deep Q-Networks (DQN)** for each agent
- **Experience replay** for stable learning
- **Cooperative reward structures**
- **Decentralized execution with centralized training**
- **Collision avoidance** between agents
- **Visualization** of training progress and agent paths

## Requirements

- Python 3.6+
- TensorFlow 2.x
- NumPy
- Matplotlib (for visualization)
- tkinter (for GUI)

## How to Use

### Standalone Visualization

1. Run the MARL_Integration.py file:
   ```
   python MARL_Integration.py
   ```

2. The visualizer will open with the following features:
   - Grid visualization
   - Controls for grid size, number of agents, and training episodes
   - Buttons for adding/clearing obstacles
   - Training controls (Start/Stop)
   - Test button to see the trained agents in action
   - Real-time metrics visualization (rewards, success rate, etc.)
   - Training log

### Integration with Your Project

To integrate the MARL system with your existing project:

1. Import the PathPlanningMARL class:
   ```python
   from PathPlanningMARL import PathPlanningMARL
   ```

2. Create an instance with your desired parameters:
   ```python
   marl = PathPlanningMARL(
       grid_size=10,  # Match your grid size
       num_agents=3,  # Number of agents to control
       gamma=0.95,    # Discount factor
       epsilon=1.0,   # Initial exploration rate
       epsilon_decay=0.995  # Exploration decay rate
   )
   ```

3. Convert your grid to the format expected by the MARL system:
   ```python
   # Convert your grid cells to the format expected by MARL
   grid_state = {}
   for cell in self.cells:
       grid_state[cell] = {
           "obstacle": self.cells[cell]["obstacle"],
           "active": self.cells[cell]["active"]
       }
   ```

4. Plan paths using the trained models:
   ```python
   # Get agent positions and target positions
   agent_positions = [pos for pos in self.active_cells]
   target_positions = [pos for pos in self.target_shape[:len(agent_positions)]]
   
   # Plan paths
   result = marl.plan_paths(
       grid_state, 
       agent_positions, 
       target_positions, 
       max_steps=100,
       training=False  # Set to True during training
   )
   
   if result:
       # Use the paths
       for i, path in enumerate(result['paths']):
           print(f"Agent {i} path: {path}")
   ```

5. Train the system:
   ```python
   # Train for 1000 episodes
   marl.train_agents(
       grid_state,
       num_episodes=1000,
       max_steps=100,
       save_interval=100,
       print_interval=10
   )
   ```

## Monitoring Training Progress

The MARL system tracks several metrics during training:

- **Episode rewards** - Average reward per episode
- **Success rate** - Percentage of agents reaching their targets
- **Episode length** - Number of steps taken per episode
- **Exploration rate (epsilon)** - Decreases over time as agents learn

These metrics are visualized in real-time in the standalone visualization tool.

## Saving and Loading Models

Models are automatically saved during training at specified intervals. They are saved to a directory called `marl_models` in the current working directory.

The models are automatically loaded when you create a new PathPlanningMARL instance, so trained agents will use their learned policies.

## Customization

You can customize various aspects of the MARL system:

- **Reward structure** - Modify the `calculate_reward` method
- **Neural network architecture** - Modify the `_build_model` method
- **State representation** - Modify the `_get_state_representation` method
- **Exploration strategy** - Adjust epsilon parameters

## Troubleshooting

- If you encounter errors about missing TensorFlow, install it with:
  ```
  pip install tensorflow
  ```

- If the visualization doesn't work, make sure you have matplotlib installed:
  ```
  pip install matplotlib
  ```

- If agents aren't learning effectively, try:
  - Increasing the number of training episodes
  - Adjusting the reward structure
  - Modifying the neural network architecture
  - Changing the exploration parameters
