# Shape Formation MARL with Collision Avoidance

This module enhances the Multi-Agent Reinforcement Learning (MARL) system to focus on two primary goals:
1. Shape formation as the ultimate objective
2. Collision avoidance during path planning

The agents are trained to cooperatively form specific shapes while ensuring they don't collide with each other during movement.

## Key Enhancements

### 1. Collision Avoidance and Shape Formation Rewards

The reward function has been enhanced to prioritize both collision avoidance and shape formation:

- **Collision Penalties**: Significant penalties for moves that would result in collisions or near-collisions
- **Path Crossing Prevention**: Penalties for agents that cross paths with other agents
- **Safe Distance Rewards**: Rewards for maintaining safe distances from other agents
- **Shape Completion Bonus**: Agents receive significant rewards when the entire shape is completed
- **Formation Error Reduction**: Rewards for reducing the overall shape error
- **Cooperative Formation**: Agents are rewarded for moves that help the overall shape formation
- **Path Clearing**: Enhanced rewards for agents that move out of the way to help other agents reach their targets

### 2. Enhanced State Representation

The state representation now includes shape-specific information:

- **Shape Error**: How well the current agent positions match the target shape
- **Shape Completion**: Percentage of targets that have been reached
- **Position in Formation**: Whether the agent's target is central or peripheral to the shape
- **Shape Outline**: The grid representation now includes information about the shape outline

### 3. Curriculum Learning

A curriculum learning approach has been implemented to gradually increase the complexity of shapes:

- **Progressive Shapes**: Training starts with simple shapes (lines) and progresses to more complex ones (circles, filled rectangles)
- **Difficulty Progression**: The distance between agents and their targets increases as training progresses
- **Success Thresholds**: Agents must achieve a certain success rate before advancing to more complex shapes

### 4. Shape Templates

Several shape templates have been implemented for training:

- **Lines**: Horizontal and vertical lines
- **Rectangles**: Outline and filled rectangles of different sizes
- **Triangles**: Simple triangular shapes
- **Circles**: Circular formations of different sizes

## Usage

To train the MARL system with shape formation as the goal:

```python
from PathPlanningMARL import PathPlanningMARL

# Initialize the MARL system
marl = PathPlanningMARL(
    grid_size=10,
    num_agents=8,
    epsilon=1.0,
    epsilon_decay=0.997,
    gamma=0.98,
    learning_rate=0.0005
)

# Train with curriculum learning
marl.train_agents(
    grid_state=grid_state,
    num_episodes=500,
    max_steps=200,
    save_interval=50,
    print_interval=10,
    curriculum_learning=True
)
```

## Integration with the Shape Formation Simulator

The enhanced MARL system can be integrated with the existing Shape Formation Simulator:

1. The MARL system will now prioritize forming the desired shape
2. Agents will cooperate to achieve the overall shape formation goal
3. The system can handle various shape types defined in the simulator

## Training Metrics

The training process now tracks shape-specific metrics:

- **Completion Rate**: Percentage of targets successfully reached
- **Formation Error**: How well the agents match the target shape
- **Curriculum Level**: Current level in the curriculum learning process

## Model Architecture

The neural network architecture has been enhanced to process shape-specific information:

- **Shape Metrics Processing**: Dedicated neural network layers for processing shape formation metrics
- **Enhanced Feature Combination**: Shape features are combined with path and agent information
- **Dueling DQN Architecture**: Maintained for stable learning with shape-specific enhancements

## Version Information

The model version has been updated to version 3 to reflect the shape formation enhancements:

- **Version 3**: Enhanced for shape formation as the ultimate goal
- **State Size**: Increased to include shape metrics
- **Architecture**: Updated to "dueling_dqn_shape_formation"

## Future Enhancements

Potential future enhancements to the shape formation MARL system:

1. **Dynamic Shape Adaptation**: Ability to adapt to changing shape requirements during training
2. **Hierarchical Formation**: Agents learn to form complex shapes by combining simpler shapes
3. **Obstacle Avoidance Optimization**: Enhanced ability to form shapes in environments with obstacles
4. **Formation Maintenance**: Ability to maintain formation while moving as a group
5. **Collision Prediction**: Advanced collision prediction to anticipate and avoid potential collisions several steps ahead
6. **Traffic Management**: Implementing traffic rules for agents to follow when paths intersect
7. **Priority-Based Movement**: Assigning priorities to agents based on their position in the shape formation
