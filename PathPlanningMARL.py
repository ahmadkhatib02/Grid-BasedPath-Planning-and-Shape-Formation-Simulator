import numpy as np
import os
import pickle
import random
import json
import time
import warnings
from collections import deque

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1=INFO, 2=WARNING, 3=ERROR)

# Force TensorFlow to use the GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow memory growth

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import (
        Dense, Dropout, Input, concatenate, Lambda,
        Add, Subtract, Layer
    )
    from tensorflow.keras.optimizers import Adam

    # Configure TensorFlow to reduce retracing warnings
    tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings

    # Configure GPU for optimal performance
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            print(f"Found {len(gpus)} GPU(s): {gpus}")
            for gpu in gpus:
                # Enable memory growth to avoid allocating all GPU memory at once
                tf.config.experimental.set_memory_growth(gpu, True)

                # Set memory limit to 80% of GPU memory to avoid OOM errors
                # Adjust this value based on your GPU memory and requirements
                # gpu_memory = tf.config.experimental.get_memory_info(gpu)['total'] * 0.8
                # tf.config.experimental.set_virtual_device_configuration(
                #     gpu,
                #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory)]
                # )

            # Log GPU information
            print("GPU is enabled for training")

            # Set TensorFlow to use the GPU
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
                print(f"Using GPU: {physical_devices[0].name}")
            else:
                print("No GPU found, falling back to CPU")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU found, falling back to CPU")

    # Verify TensorFlow is using GPU
    print(f"TensorFlow is using GPU: {tf.test.is_gpu_available()}")
    print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")

    ML_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. MARL features will be disabled.")
    ML_AVAILABLE = False

class PathPlanningMARL:
    """
    Multi-Agent Reinforcement Learning for path planning with obstacles.

    This class implements:
    1. Deep Q-Networks (DQN) for each agent
    2. Experience replay for stable learning
    3. Cooperative reward structures
    4. Decentralized execution with centralized training
    5. Collision avoidance between agents
    """

    def __init__(self, grid_size=10, num_agents=2, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001,
                 memory_size=10000, batch_size=64):
        """
        Initialize the PathPlanningMARL class.

        Args:
            grid_size: Size of the grid (default: 10x10)
            num_agents: Number of agents in the environment
            gamma: Discount factor for future rewards
            epsilon: Exploration rate (probability of taking a random action)
            epsilon_min: Minimum exploration rate
            epsilon_decay: Rate at which exploration decreases
            learning_rate: Learning rate for the neural network
            memory_size: Size of the experience replay memory
            batch_size: Batch size for training
        """
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size

        # Action space: up, right, down, left
        self.action_size = 4
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left

        # Experience replay memory for each agent
        self.memories = [deque(maxlen=memory_size) for _ in range(num_agents)]

        # Create models for each agent
        if ML_AVAILABLE:
            self.models = [self._build_model() for _ in range(num_agents)]
            self.target_models = [self._build_model() for _ in range(num_agents)]
            self._update_target_models()
        else:
            self.models = None
            self.target_models = None

        # Model file paths
        self.model_dir = "marl_models"
        if not os.path.exists(self.model_dir) and ML_AVAILABLE:
            os.makedirs(self.model_dir)

        # Load existing models if available
        self.load_models()

        # Training metrics
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rate': []
        }

    def _build_model(self):
        """
        Build an advanced Deep Q-Network model optimized for shape formation.
        Enhanced to process shape-specific information and formation metrics.
        Includes optimizations to reduce TensorFlow retracing.
        Explicitly configured to use GPU for training.

        Returns:
            Keras model
        """
        if not ML_AVAILABLE:
            return None

        # Explicitly use GPU for model building and training
        with tf.device('/GPU:0'):
            print("Building model on GPU...")

            # Enhanced input structure:
            # 2 (agent pos) + 2 (target pos) + 3 (path info) + 12 (nearest agents) +
            # 5 (shape metrics) + grid_size*grid_size (grid)
            input_size = self.grid_size * self.grid_size + 24  # Added 5 for shape metrics
            state_input = Input(shape=(input_size,), dtype=tf.float32, name="state_input")

            # Add batch normalization to standardize inputs and reduce retracing
            normalized_input = tf.keras.layers.BatchNormalization()(state_input)

            # Split the input into different components
            # Agent and target positions (first 4 values)
            positions = Lambda(lambda x: x[:, :4], name="positions_extractor")(normalized_input)

            # Path information (next 3 values)
            path_info = Lambda(lambda x: x[:, 4:7], name="path_info_extractor")(normalized_input)

            # Nearest agents with detailed info (next 12 values)
            nearest_agents = Lambda(lambda x: x[:, 7:19], name="agents_info_extractor")(normalized_input)

            # Shape formation metrics (next 5 values)
            shape_metrics = Lambda(lambda x: x[:, 19:24], name="shape_metrics_extractor")(normalized_input)

            # Grid state (remaining values)
            grid = Lambda(lambda x: x[:, 24:], name="grid_extractor")(normalized_input)

            # Process positions with deeper network
            pos_features = Dense(24, activation='relu', name="pos_dense1")(positions)
            pos_features = Dense(16, activation='relu', name="pos_dense2")(pos_features)

            # Process path information
            path_features = Dense(16, activation='relu', name="path_dense1")(path_info)
            path_features = Dense(12, activation='relu', name="path_dense2")(path_features)

            # Process nearest agents with specialized layers
            agent_features = Dense(32, activation='relu', name="agent_dense1")(nearest_agents)
            agent_features = Dropout(0.1, name="agent_dropout1")(agent_features)
            agent_features = Dense(24, activation='relu', name="agent_dense2")(agent_features)

            # Process shape metrics with specialized layers
            # This is crucial for shape formation as the ultimate goal
            shape_features = Dense(20, activation='relu', name="shape_dense1")(shape_metrics)
            shape_features = Dense(16, activation='relu', name="shape_dense2")(shape_features)

            # Process grid with attention to path blocking and shape formation
            grid_features = Dense(128, activation='relu', name="grid_dense1")(grid)
            grid_features = Dropout(0.2, name="grid_dropout1")(grid_features)
            grid_features = Dense(96, activation='relu', name="grid_dense2")(grid_features)
            grid_features = Dropout(0.2, name="grid_dropout2")(grid_features)
            grid_features = Dense(64, activation='relu', name="grid_dense3")(grid_features)

            # Combine all features
            combined = concatenate([
                pos_features,
                path_features,
                agent_features,
                shape_features,  # Add shape features to the combined representation
                grid_features
            ], name="feature_concat")

            # Add batch normalization after concatenation to stabilize training
            combined = tf.keras.layers.BatchNormalization(name="combined_batchnorm")(combined)

            # Shared feature extraction with deeper network
            x = Dense(128, activation='relu', name="shared_dense1")(combined)
            x = Dropout(0.2, name="shared_dropout1")(x)
            x = Dense(96, activation='relu', name="shared_dense2")(x)
            x = Dropout(0.2, name="shared_dropout2")(x)
            x = Dense(64, activation='relu', name="shared_dense3")(x)
            x = Dropout(0.1, name="shared_dropout3")(x)
            x = Dense(32, activation='relu', name="shared_dense4")(x)

            # Advantage and value streams (Dueling DQN architecture)
            advantage_stream = Dense(32, activation='relu', name="advantage_dense")(x)
            advantage = Dense(self.action_size, activation='linear', name="advantage_output")(advantage_stream)

            value_stream = Dense(32, activation='relu', name="value_dense")(x)
            value = Dense(1, activation='linear', name="value_output")(value_stream)

            # Combine streams to get Q-values
            mean_advantage = Lambda(lambda x: K.mean(x, axis=1, keepdims=True), name="advantage_mean")(advantage)
            q_values = Add(name="q_values")([value, Subtract(name="advantage_subtract")([advantage, mean_advantage])])

            # Use Huber loss for better stability with outliers
            model = Model(inputs=state_input, outputs=q_values)

            # Configure model for reduced retracing and GPU optimization
            # Create optimizer with explicit configuration
            optimizer = Adam(
                learning_rate=self.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                clipnorm=1.0,
                name='adam'
            )

            # Use mixed precision for faster GPU training
            try:
                # Enable mixed precision training (FP16) for faster GPU computation
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("Mixed precision training enabled (FP16)")
            except Exception as e:
                print(f"Could not enable mixed precision: {e}")

            # Use a custom compile configuration to reduce retracing
            model.compile(
                loss=tf.keras.losses.Huber(),  # Use object instead of string
                optimizer=optimizer,
                # Disable run_eagerly for better performance
                run_eagerly=False
            )

            # Create a dummy input to initialize the model and optimizer
            dummy_input = np.zeros((1, input_size), dtype=np.float32)
            model.predict(dummy_input)  # Use predict instead of direct call to ensure full initialization

            # Ensure optimizer is initialized
            dummy_target = np.zeros((1, self.action_size), dtype=np.float32)
            model.train_on_batch(dummy_input, dummy_target)

            # Print model summary
            model.summary()

            # Log that the model is built on GPU
            print(f"Model built successfully on GPU. Input shape: {input_size}, Output shape: {self.action_size}")

        return model

    def _update_target_models(self):
        """Update target models to match the primary models."""
        if not ML_AVAILABLE:
            return

        for i in range(self.num_agents):
            self.target_models[i].set_weights(self.models[i].get_weights())

    def _get_state_representation(self, grid_state, agent_pos, target_pos, other_agent_positions,
                                 all_target_positions=None):
        """
        Enhanced state representation optimized for shape formation as the ultimate goal.
        Includes shape-specific information and formation metrics.
        Optimized to reduce TensorFlow retracing.

        Args:
            grid_state: Dictionary of cell states
            agent_pos: Position of the agent (row, col)
            target_pos: Target position (row, col)
            other_agent_positions: Positions of other agents
            all_target_positions: All target positions (for better coordination)

        Returns:
            numpy array representing the state
        """
        if not ML_AVAILABLE:
            return None

        # Create a flattened grid representation with more detailed cell states
        # 0: Empty cell
        # 1: Obstacle
        # 2: Other agent
        # 3: Other target
        # 4: Potential path cell (cells on direct path to target)
        # 5: Blocked path cell (cells on path that are blocked by other agents)
        # 6: Shape outline cell (cells that form the outline of the target shape)
        grid_flat = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)

        # Mark obstacles
        for pos, cell in grid_state.items():
            if cell.get("obstacle", False):
                idx = pos[0] * self.grid_size + pos[1]
                if 0 <= idx < len(grid_flat):  # Ensure index is valid
                    grid_flat[idx] = 1.0

        # Calculate direct path to target (for path blocking detection)
        direct_path_cells = self._get_direct_path_cells(agent_pos, target_pos)
        direct_path_set = set(direct_path_cells)  # Convert to set for faster lookups

        # Mark potential path cells
        for pos in direct_path_cells:
            if pos != agent_pos and pos != target_pos:  # Don't mark agent or target
                idx = pos[0] * self.grid_size + pos[1]
                if 0 <= idx < len(grid_flat) and grid_flat[idx] == 0:  # Only mark if cell is empty
                    grid_flat[idx] = 4.0

        # Mark other agents' positions and detect blocked paths
        other_agent_set = set(other_agent_positions)
        for pos in other_agent_positions:
            if pos != agent_pos:  # Don't mark the current agent
                idx = pos[0] * self.grid_size + pos[1]
                if 0 <= idx < len(grid_flat):
                    grid_flat[idx] = 2.0  # Mark as other agent

                    # If this agent is on our direct path, mark it as a blocked path cell
                    if pos in direct_path_set:
                        grid_flat[idx] = 5.0  # Mark as blocked path cell

        # Mark target positions in the grid
        if all_target_positions:
            # Create a set of all target positions for faster lookups
            all_targets_set = set(all_target_positions)

            # Mark all target positions
            for pos in all_target_positions:
                if pos != target_pos:  # Don't double-mark the current agent's target
                    idx = pos[0] * self.grid_size + pos[1]
                    if 0 <= idx < len(grid_flat):
                        # If a cell is already marked as an agent, keep it as an agent
                        if grid_flat[idx] != 2.0 and grid_flat[idx] != 5.0:
                            grid_flat[idx] = 3.0  # Use 3 to represent other targets

            # Mark shape outline cells - cells that form the boundary of the target shape
            # This helps agents understand the overall shape they're trying to form
            if len(all_target_positions) >= 3:  # Need at least 3 points to form a shape
                # Find the bounding box of the target shape
                min_row = min(pos[0] for pos in all_target_positions)
                max_row = max(pos[0] for pos in all_target_positions)
                min_col = min(pos[1] for pos in all_target_positions)
                max_col = max(pos[1] for pos in all_target_positions)

                # Mark cells on the perimeter of the bounding box that aren't targets
                for r in range(min_row, max_row + 1):
                    for c in range(min_col, max_col + 1):
                        # Only mark cells on the perimeter
                        if r == min_row or r == max_row or c == min_col or c == max_col:
                            pos = (r, c)
                            idx = r * self.grid_size + c

                            # Only mark if not already a target and within grid bounds
                            if (pos not in all_targets_set and
                                0 <= idx < len(grid_flat) and
                                grid_flat[idx] == 0):
                                grid_flat[idx] = 6.0  # Mark as shape outline

        # Calculate relative positions and states of nearest agents
        # Pre-allocate array with zeros for better performance
        nearest_agents_info = np.zeros(12, dtype=np.float32)  # 4 values per agent, 3 agents

        if other_agent_positions:
            # Calculate distances for sorting
            distances = [(pos, abs(pos[0] - agent_pos[0]) + abs(pos[1] - agent_pos[1]))
                         for pos in other_agent_positions if pos != agent_pos]

            # Sort by distance
            distances.sort(key=lambda x: x[1])

            # Take up to 3 nearest agents
            for i in range(min(3, len(distances))):
                agent_pos_other, _ = distances[i]
                idx = i * 4  # Index in the nearest_agents_info array

                # Calculate relative position (normalized)
                nearest_agents_info[idx] = (agent_pos_other[0] - agent_pos[0]) / self.grid_size
                nearest_agents_info[idx + 1] = (agent_pos_other[1] - agent_pos[1]) / self.grid_size

                # Calculate if this agent is blocking our path
                nearest_agents_info[idx + 2] = 1.0 if agent_pos_other in direct_path_set else 0.0

                # Calculate if we're blocking this agent's path (if we have target info)
                if all_target_positions and i < len(all_target_positions):
                    other_target = all_target_positions[i]
                    other_path = self._get_direct_path_cells(agent_pos_other, other_target)
                    nearest_agents_info[idx + 3] = 1.0 if agent_pos in other_path else 0.0

        # Calculate distance to target and path complexity
        distance_to_target = abs(agent_pos[0] - target_pos[0]) + abs(agent_pos[1] - target_pos[1])
        path_complexity = len(direct_path_cells) / max(1, distance_to_target)  # Avoid division by zero

        # Calculate corridor detection (is agent in a tight space?)
        in_corridor = 1.0 if self._is_in_corridor(agent_pos, other_agent_set) else 0.0

        # SHAPE FORMATION METRICS
        # Calculate shape-specific information to help with formation
        shape_metrics = np.zeros(5, dtype=np.float32)

        if all_target_positions:
            # 1. Calculate current shape error
            all_agent_positions = list(other_agent_positions) + [agent_pos]
            current_shape_error = self._calculate_shape_error(all_agent_positions, all_target_positions)
            shape_metrics[0] = min(1.0, current_shape_error / (self.grid_size * 2))  # Normalize

            # 2. Calculate shape completion percentage
            if all_target_positions:
                # Count how many targets have agents at or near them
                targets_filled = 0
                for target in all_target_positions:
                    for agent in all_agent_positions:
                        if abs(agent[0] - target[0]) + abs(agent[1] - target[1]) <= 1:
                            targets_filled += 1
                            break

                shape_completion = targets_filled / len(all_target_positions)
                shape_metrics[1] = shape_completion

            # 3. Calculate agent's position in the formation (center vs. edge)
            if len(all_target_positions) >= 3:
                # Calculate centroids
                target_centroid = self._calculate_centroid(all_target_positions)

                # Calculate distance from agent's target to centroid
                target_to_centroid = abs(target_pos[0] - target_centroid[0]) + abs(target_pos[1] - target_centroid[1])

                # Normalize by maximum possible distance in the grid
                max_distance = self.grid_size * 2
                shape_metrics[2] = target_to_centroid / max_distance

                # 4. Is the agent's target on the perimeter of the shape?
                min_row = min(pos[0] for pos in all_target_positions)
                max_row = max(pos[0] for pos in all_target_positions)
                min_col = min(pos[1] for pos in all_target_positions)
                max_col = max(pos[1] for pos in all_target_positions)

                is_perimeter = (target_pos[0] == min_row or target_pos[0] == max_row or
                               target_pos[1] == min_col or target_pos[1] == max_col)
                shape_metrics[3] = 1.0 if is_perimeter else 0.0

                # 5. How many neighbors does this target have in the shape?
                neighbor_count = 0
                for other_target in all_target_positions:
                    if other_target != target_pos:
                        if abs(other_target[0] - target_pos[0]) + abs(other_target[1] - target_pos[1]) == 1:
                            neighbor_count += 1

                shape_metrics[4] = min(1.0, neighbor_count / 4.0)  # Normalize (max 4 neighbors)

        # Pre-allocate arrays for better performance
        agent_pos_norm = np.array([agent_pos[0] / self.grid_size, agent_pos[1] / self.grid_size], dtype=np.float32)
        target_pos_norm = np.array([target_pos[0] / self.grid_size, target_pos[1] / self.grid_size], dtype=np.float32)
        path_info = np.array([distance_to_target / self.grid_size, path_complexity, in_corridor], dtype=np.float32)

        # Combine all information into the state representation
        # Use np.concatenate with dtype specified for consistent output
        state = np.concatenate([
            agent_pos_norm,
            target_pos_norm,
            path_info,
            nearest_agents_info,
            shape_metrics,  # Add shape formation metrics
            grid_flat
        ]).astype(np.float32)

        return state

    def _get_direct_path_cells(self, start, end):
        """
        Get cells on the direct path from start to end.

        Args:
            start: Start position
            end: End position

        Returns:
            List of positions on the direct path
        """
        path_cells = []

        # Calculate differences and steps
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        # Handle straight lines first (simpler)
        if dx == 0:  # Vertical line
            step = 1 if dy > 0 else -1
            for y in range(start[1], end[1] + step, step):
                path_cells.append((start[0], y))
            return path_cells

        if dy == 0:  # Horizontal line
            step = 1 if dx > 0 else -1
            for x in range(start[0], end[0] + step, step):
                path_cells.append((x, start[1]))
            return path_cells

        # Handle diagonal lines
        if abs(dx) == abs(dy):  # Perfect diagonal
            step_x = 1 if dx > 0 else -1
            step_y = 1 if dy > 0 else -1
            for i in range(abs(dx) + 1):
                path_cells.append((start[0] + i * step_x, start[1] + i * step_y))
            return path_cells

        # For non-direct paths, use Bresenham's line algorithm
        # This is a simplified version that works well for grid-based movement
        path_cells = [start]  # Start with the start position

        # Determine primary direction and steps
        if abs(dx) > abs(dy):  # Horizontal movement is primary
            step_x = 1 if dx > 0 else -1
            slope = abs(dy / dx)

            for i in range(1, abs(dx) + 1):
                x = start[0] + i * step_x
                y_float = start[1] + i * slope * (1 if dy > 0 else -1)
                y = round(y_float)
                path_cells.append((x, y))

        else:  # Vertical movement is primary
            step_y = 1 if dy > 0 else -1
            slope = abs(dx / dy)

            for i in range(1, abs(dy) + 1):
                y = start[1] + i * step_y
                x_float = start[0] + i * slope * (1 if dx > 0 else -1)
                x = round(x_float)
                path_cells.append((x, y))

        return path_cells

    def remember(self, agent_idx, state, action, reward, next_state, done):
        """
        Store experience in the replay memory.

        Args:
            agent_idx: Index of the agent
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        if not ML_AVAILABLE:
            return

        self.memories[agent_idx].append((state, action, reward, next_state, done))

    # Define a TensorFlow function for prediction to avoid retracing
    # Use input_signature to further reduce retracing
    @tf.function(
        experimental_relax_shapes=True,
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float32),  # model is not a tensor
            tf.TensorSpec(shape=[None, None], dtype=tf.float32)  # state_tensor
        ]
    )
    def _predict_q_values(self, model, state_tensor):
        """
        TensorFlow function for predicting Q-values to avoid retracing.

        Args:
            model: The model to use for prediction
            state_tensor: Input state tensor

        Returns:
            Predicted Q-values
        """
        # Explicitly set training=False to avoid batch normalization issues
        return model(state_tensor, training=False)

    def act(self, agent_idx, state, valid_actions, training=True):
        """
        Choose an action for an agent using epsilon-greedy policy.
        Optimized to reduce TensorFlow retracing.

        Args:
            agent_idx: Index of the agent
            state: Current state
            valid_actions: List of valid action indices
            training: Whether we're in training mode (affects exploration)

        Returns:
            Chosen action index
        """
        if not ML_AVAILABLE or not valid_actions:
            return random.choice(valid_actions) if valid_actions else None

        # Exploration: choose a random action
        if training and np.random.rand() < self.epsilon:
            return random.choice(valid_actions)

        # Ensure state has the correct shape and type
        # This helps prevent TensorFlow retracing
        state_tensor = np.asarray(state, dtype=np.float32).reshape(1, -1)

        # Use predict_on_batch for single sample prediction
        # This is more reliable than using the TensorFlow function directly
        q_values = self.models[agent_idx].predict_on_batch(state_tensor)[0]

        # Filter for valid actions only
        valid_q = [(i, q_values[i]) for i in valid_actions]
        valid_q.sort(key=lambda x: x[1], reverse=True)

        return valid_q[0][0]  # Return the action with highest Q-value

    def get_valid_actions(self, grid_state, agent_pos, all_agent_positions):
        """
        Get valid actions for an agent (no obstacles, within bounds, no collisions).

        Args:
            grid_state: Dictionary of cell states
            agent_pos: Position of the agent
            all_agent_positions: Positions of all agents

        Returns:
            List of valid action indices
        """
        valid_actions = []

        for i, (dr, dc) in enumerate(self.actions):
            new_r, new_c = agent_pos[0] + dr, agent_pos[1] + dc
            new_pos = (new_r, new_c)

            # Check if the move is valid (within bounds, no obstacle)
            if (0 <= new_r < self.grid_size and
                0 <= new_c < self.grid_size and
                not grid_state.get(new_pos, {}).get("obstacle", False)):

                # Check for collision with other agents
                if new_pos not in all_agent_positions:
                    valid_actions.append(i)

        return valid_actions

    def calculate_reward(self, agent_pos, new_pos, target_pos, done, all_agent_positions=None,
                         all_target_positions=None, all_dones=None, step_penalty=-0.1,
                         collision_penalty=-2.5, goal_reward=15.0, proximity_factor=0.15,
                         cooperative_factor=0.6, formation_factor=0.5, path_clearing_factor=0.6,
                         shape_completion_factor=1.0, collision_avoidance_factor=0.8):
        """
        Enhanced reward function optimized for shape formation and collision avoidance as the ultimate goals.

        Args:
            agent_pos: Current position
            new_pos: New position after action
            target_pos: Target position
            done: Whether the target is reached
            all_agent_positions: Positions of all agents (for cooperative rewards)
            all_target_positions: Target positions of all agents (for formation rewards)
            all_dones: List of done flags for all agents
            step_penalty: Penalty for each step
            collision_penalty: Penalty for collisions
            goal_reward: Reward for reaching the goal
            proximity_factor: Factor for rewarding getting closer to the goal
            cooperative_factor: Factor for cooperative rewards
            formation_factor: Factor for maintaining formation
            path_clearing_factor: Factor for rewarding path clearing behavior
            shape_completion_factor: Factor for rewarding shape completion
            collision_avoidance_factor: Factor for rewarding collision avoidance

        Returns:
            Calculated reward
        """
        # Base reward calculation
        base_reward = 0

        # Goal reached - enhanced reward based on how many other agents have reached their goals
        # and how well the overall shape is forming
        if done:
            # Base goal reward
            base_reward = goal_reward

            # Additional reward if other agents have also reached their goals
            if all_dones:
                completed_count = sum(1 for d in all_dones if d)
                # Bonus increases with more agents completing their targets
                completion_bonus = (completed_count / len(all_dones)) * goal_reward * 0.8
                base_reward += completion_bonus

                # Extra bonus for completing the entire shape (all agents at targets)
                if completed_count == len(all_dones):
                    base_reward += goal_reward * 0.7  # Significant bonus for full shape completion

                # Partial shape completion bonus - more agents in position means better shape
                shape_completion_ratio = completed_count / len(all_dones)
                shape_bonus = shape_completion_ratio * shape_completion_ratio * goal_reward * 0.5
                base_reward += shape_bonus

            return base_reward

        # Invalid move (collision or out of bounds) - significantly increased penalty
        if new_pos == agent_pos:
            return collision_penalty

        # Calculate distance change to target
        old_distance = abs(agent_pos[0] - target_pos[0]) + abs(agent_pos[1] - target_pos[1])
        new_distance = abs(new_pos[0] - target_pos[0]) + abs(new_pos[1] - target_pos[1])
        distance_change = old_distance - new_distance

        # Reward for getting closer to target, penalty for moving away
        proximity_reward = distance_change * proximity_factor
        base_reward += proximity_reward

        # Cooperative rewards (if we have information about other agents)
        if all_agent_positions and all_target_positions:
            # COLLISION AVOIDANCE REWARDS - Enhanced to prioritize avoiding collisions
            collision_avoidance_reward = 0

            # Check for potential collisions with other agents
            for other_pos in all_agent_positions:
                if other_pos != agent_pos:
                    # Calculate current distance to other agent
                    current_dist = abs(agent_pos[0] - other_pos[0]) + abs(agent_pos[1] - other_pos[1])

                    # Calculate new distance to other agent after move
                    new_dist = abs(new_pos[0] - other_pos[0]) + abs(new_pos[1] - other_pos[1])

                    # Significant penalty for moving too close to another agent
                    if new_dist == 0:  # Direct collision
                        collision_avoidance_reward -= 2.0
                    elif new_dist == 1:  # Adjacent (too close)
                        collision_avoidance_reward -= 0.5

                    # Reward for maintaining safe distance
                    if current_dist <= 1 and new_dist > 1:
                        collision_avoidance_reward += 1.0  # Reward for moving away from collision

                    # Check for crossing paths (agents swapping positions)
                    if new_pos == other_pos and agent_pos in [pos for pos in all_agent_positions if pos != agent_pos]:
                        collision_avoidance_reward -= 1.5  # Penalty for path crossing

            # Add collision avoidance reward
            base_reward += collision_avoidance_reward * collision_avoidance_factor

            # Path blocking detection and rewards
            path_blocking_reward = 0

            # Check if this agent is potentially blocking others' paths
            for i, other_agent_pos in enumerate(all_agent_positions):
                if other_agent_pos != agent_pos and i < len(all_target_positions):
                    other_target = all_target_positions[i]

                    # Check if agent is between other agent and its target
                    is_between_before = self._is_between(agent_pos, other_agent_pos, other_target)
                    is_between_after = self._is_between(new_pos, other_agent_pos, other_target)

                    # Reward for moving out of the way (clearing path)
                    if is_between_before and not is_between_after:
                        path_blocking_reward += 1.0  # Increased reward for clearing paths
                    # Penalty for moving into the way (blocking path)
                    elif not is_between_before and is_between_after:
                        path_blocking_reward -= 1.0  # Increased penalty for blocking paths

                    # Check if agent is moving to make room in tight corridors
                    if self._is_in_corridor(agent_pos, all_agent_positions) and not self._is_in_corridor(new_pos, all_agent_positions):
                        path_blocking_reward += 0.8  # Increased reward for clearing corridors

            # Add path blocking reward
            base_reward += path_blocking_reward * path_clearing_factor

            # SHAPE FORMATION REWARDS - Enhanced to prioritize overall shape
            # Calculate current shape metrics
            current_shape_error = self._calculate_shape_error(all_agent_positions, all_target_positions)

            # Calculate what the shape error would be after this move
            new_agent_positions = all_agent_positions.copy()
            for i, pos in enumerate(new_agent_positions):
                if pos == agent_pos:
                    new_agent_positions[i] = new_pos
                    break

            new_shape_error = self._calculate_shape_error(new_agent_positions, all_target_positions)

            # Reward for improving the overall shape
            shape_improvement = current_shape_error - new_shape_error
            shape_reward = 0

            if shape_improvement > 0:
                # Significant reward for moves that improve the overall shape
                shape_reward += shape_improvement * 1.8
            elif shape_improvement < 0:
                # Penalty for moves that worsen the overall shape
                shape_reward += shape_improvement * 1.0

            # Add shape formation reward
            base_reward += shape_reward * shape_completion_factor

            # Reward for maintaining appropriate distance from other agents
            # (not too close to cause congestion, not too far to break formation)
            formation_reward = 0

            # Count nearby agents
            nearby_agents = 0
            for other_pos in all_agent_positions:
                if other_pos != agent_pos:
                    # Calculate Manhattan distance to other agent
                    dist = abs(new_pos[0] - other_pos[0]) + abs(new_pos[1] - other_pos[1])

                    # Penalize being too close (congestion)
                    if dist < 2:
                        formation_reward -= 0.3  # Increased penalty for being too close
                    # Reward being at a good distance (coordination)
                    elif 2 <= dist <= 3:
                        formation_reward += 0.2  # Increased reward for good spacing
                        nearby_agents += 1

            # Bonus for being in a good formation position (having the right number of neighbors)
            if 1 <= nearby_agents <= 2:
                formation_reward += 0.3  # Increased bonus for good formation position

            # Add formation reward to base reward
            base_reward += formation_reward * formation_factor

            # Reward for moving in a way that helps the overall shape formation
            # Calculate how well the current agent positions match the target formation
            current_formation_error = 0
            for i, agent in enumerate(all_agent_positions):
                if i < len(all_target_positions):
                    current_formation_error += abs(agent[0] - all_target_positions[i][0]) + abs(agent[1] - all_target_positions[i][1])

            # Calculate what the formation error would be after this move
            new_formation_error = current_formation_error
            for i, agent in enumerate(all_agent_positions):
                if agent == agent_pos and i < len(all_target_positions):
                    # Replace this agent's position with the new position
                    new_formation_error -= abs(agent[0] - all_target_positions[i][0]) + abs(agent[1] - all_target_positions[i][1])
                    new_formation_error += abs(new_pos[0] - all_target_positions[i][0]) + abs(new_pos[1] - all_target_positions[i][1])

            # Reward for reducing overall formation error
            formation_improvement = current_formation_error - new_formation_error
            if formation_improvement > 0:
                base_reward += formation_improvement * cooperative_factor

            # Special reward for waiting when blocked
            # If agent is close to target but can't move directly to it due to blockage
            if old_distance <= 3 and new_pos == agent_pos:
                # Check if there's another agent in the direct path
                direct_path_blocked = False
                for other_pos in all_agent_positions:
                    if other_pos != agent_pos and self._is_between(other_pos, agent_pos, target_pos):
                        direct_path_blocked = True
                        break

                # If blocked, give a small positive reward for waiting
                # This prevents thrashing behavior when blocked
                if direct_path_blocked:
                    base_reward += 0.2  # Increased reward for patient waiting

        # Step penalty to encourage shorter paths
        base_reward += step_penalty

        return base_reward

    def _calculate_shape_error(self, agent_positions, target_positions):
        """
        Calculate how well the current agent positions match the target shape.
        Lower error means better shape formation.

        Args:
            agent_positions: Current positions of all agents
            target_positions: Target positions for all agents

        Returns:
            Shape error metric (lower is better)
        """
        if not agent_positions or not target_positions:
            return float('inf')

        # Calculate total Manhattan distance error
        total_error = 0

        # Calculate centroid of both agent and target positions
        agent_centroid = self._calculate_centroid(agent_positions)
        target_centroid = self._calculate_centroid(target_positions)

        # Calculate relative positions to centroid
        agent_relative = [(pos[0] - agent_centroid[0], pos[1] - agent_centroid[1])
                          for pos in agent_positions]
        target_relative = [(pos[0] - target_centroid[0], pos[1] - target_centroid[1])
                           for pos in target_positions]

        # Calculate shape error using relative positions
        # This focuses on the shape rather than absolute positions
        min_agents = min(len(agent_positions), len(target_positions))
        for i in range(min_agents):
            # Find the closest target position for each agent position
            min_dist = float('inf')
            for j in range(min_agents):
                dist = abs(agent_relative[i][0] - target_relative[j][0]) + \
                       abs(agent_relative[i][1] - target_relative[j][1])
                min_dist = min(min_dist, dist)

            total_error += min_dist

        # Add penalty for different number of agents and targets
        size_diff = abs(len(agent_positions) - len(target_positions))
        total_error += size_diff * 5  # Significant penalty for missing agents

        return total_error

    def _calculate_centroid(self, positions):
        """
        Calculate the centroid (average position) of a set of positions.

        Args:
            positions: List of (row, col) positions

        Returns:
            (avg_row, avg_col) centroid position
        """
        if not positions:
            return (0, 0)

        sum_row = sum(pos[0] for pos in positions)
        sum_col = sum(pos[1] for pos in positions)

        return (sum_row / len(positions), sum_col / len(positions))

    def _is_between(self, pos, start, end):
        """
        Check if a position is between start and end positions (in the path).

        Args:
            pos: Position to check
            start: Start position
            end: End position

        Returns:
            True if pos is between start and end, False otherwise
        """
        # Check if pos is on the direct line between start and end
        # and within the bounding box

        # Check if pos is within the bounding box of start and end
        min_row = min(start[0], end[0])
        max_row = max(start[0], end[0])
        min_col = min(start[1], end[1])
        max_col = max(start[1], end[1])

        if not (min_row <= pos[0] <= max_row and min_col <= pos[1] <= max_col):
            return False

        # For simplicity, check if pos is on the Manhattan path between start and end
        # This is a simplification but works well for grid-based movement

        # Check if pos is on a direct horizontal or vertical line
        if pos[0] == start[0] or pos[1] == start[1]:
            # Check if pos is between start and end on this line
            if pos[0] == start[0]:  # Horizontal line
                return min_col <= pos[1] <= max_col
            else:  # Vertical line
                return min_row <= pos[0] <= max_row

        # Check if pos is on a diagonal line
        row_diff = abs(end[0] - start[0])
        col_diff = abs(end[1] - start[1])

        if row_diff == col_diff:  # Perfect diagonal
            # Check if pos is on this diagonal
            pos_row_diff = abs(pos[0] - start[0])
            pos_col_diff = abs(pos[1] - start[1])

            return pos_row_diff == pos_col_diff and pos_row_diff <= row_diff

        # For non-direct paths, check if pos is on any reasonable path
        # Calculate Manhattan distance from start to end
        total_dist = abs(end[0] - start[0]) + abs(end[1] - start[1])

        # Calculate Manhattan distances from start to pos and from pos to end
        dist_through_pos = abs(pos[0] - start[0]) + abs(pos[1] - start[1]) + abs(end[0] - pos[0]) + abs(end[1] - pos[1])

        # If the distance through pos is equal to the direct distance,
        # pos is on a Manhattan path from start to end
        return dist_through_pos <= total_dist + 2  # Allow small deviation

    def _is_in_corridor(self, pos, all_positions):
        """
        Check if a position is in a corridor (has agents or obstacles on multiple sides).
        Optimized to handle both lists and sets for better performance.

        Args:
            pos: Position to check
            all_positions: All agent positions (can be a list or set)

        Returns:
            True if pos is in a corridor, False otherwise
        """
        # Check adjacent positions (up, right, down, left)
        adjacent_positions = [
            (pos[0] - 1, pos[1]),
            (pos[0], pos[1] + 1),
            (pos[0] + 1, pos[1]),
            (pos[0], pos[1] - 1)
        ]

        # Count how many adjacent positions are occupied
        # Convert all_positions to a set if it's not already one for faster lookups
        if not isinstance(all_positions, set):
            all_positions_set = set(all_positions)
        else:
            all_positions_set = all_positions

        occupied_count = sum(1 for adj_pos in adjacent_positions if adj_pos in all_positions_set)

        # If 2 or more adjacent positions are occupied, it's a corridor
        return occupied_count >= 2

    # Define TensorFlow functions for target prediction to avoid retracing
    # Use input_signature to further reduce retracing
    @tf.function(
        experimental_relax_shapes=True,
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float32),  # target_model is not a tensor
            tf.TensorSpec(shape=[None, None], dtype=tf.float32)  # next_states
        ]
    )
    def _predict_target_q_values(self, target_model, next_states):
        """
        TensorFlow function for predicting target Q-values to avoid retracing.

        Args:
            target_model: The target model to use for prediction
            next_states: Batch of next state tensors

        Returns:
            Predicted target Q-values
        """
        # Explicitly set training=False to avoid batch normalization issues
        return target_model(next_states, training=False)

    # We'll use the built-in train_on_batch method instead of a custom TensorFlow function
    # This avoids the variable creation issue inside tf.function

    def train(self):
        """
        Train all agents using experience replay.
        Optimized to reduce TensorFlow retracing.
        Explicitly configured to use GPU for training.

        Returns:
            Average loss across all agents
        """
        if not ML_AVAILABLE:
            return None

        # Explicitly use GPU for training
        with tf.device('/GPU:0'):
            print("Training on GPU...")

            # Log GPU usage at the start of training
            try:
                # Get GPU memory usage
                gpu_devices = tf.config.list_physical_devices('GPU')
                if gpu_devices:
                    print(f"Training on GPU: {gpu_devices[0].name}")
            except Exception as e:
                print(f"Could not get GPU info: {e}")

            losses = []

            for agent_idx in range(self.num_agents):
                # Skip if not enough experiences
                if len(self.memories[agent_idx]) < self.batch_size:
                    continue

                # Sample a batch of experiences
                minibatch = random.sample(self.memories[agent_idx], self.batch_size)

                # Convert to numpy arrays with consistent types to reduce retracing
                states = np.asarray([experience[0] for experience in minibatch], dtype=np.float32)
                actions = np.asarray([experience[1] for experience in minibatch], dtype=np.int32)
                rewards = np.asarray([experience[2] for experience in minibatch], dtype=np.float32)
                next_states = np.asarray([experience[3] for experience in minibatch], dtype=np.float32)
                dones = np.asarray([experience[4] for experience in minibatch], dtype=np.float32)

                # Convert to TensorFlow tensors for GPU processing
                tf_states = tf.convert_to_tensor(states, dtype=tf.float32)
                tf_next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

                # Get current Q-values using predict_on_batch for better GPU performance
                current_q_values = self.models[agent_idx].predict_on_batch(tf_states)

                # Create a copy of current Q-values to update
                targets = np.copy(current_q_values)

                # Get next Q-values using target network with predict_on_batch
                next_q_values = self.target_models[agent_idx].predict_on_batch(tf_next_states)

                # Create a mask for terminal states
                terminal_mask = dones.astype(bool)
                non_terminal_mask = ~terminal_mask

                # Vectorized update for non-terminal states
                if np.any(non_terminal_mask):
                    # Get max Q-values for next states
                    max_next_q = np.max(next_q_values[non_terminal_mask], axis=1)

                    # Create indices for the specific Q-values to update
                    batch_indices = np.arange(self.batch_size)[non_terminal_mask]
                    action_indices = actions[non_terminal_mask]

                    # Update Q-values for non-terminal states (vectorized)
                    targets[batch_indices, action_indices] = rewards[batch_indices] + self.gamma * max_next_q

                # Vectorized update for terminal states
                if np.any(terminal_mask):
                    # Create indices for the specific Q-values to update
                    batch_indices = np.arange(self.batch_size)[terminal_mask]
                    action_indices = actions[terminal_mask]

                    # Update Q-values for terminal states (vectorized)
                    targets[batch_indices, action_indices] = rewards[batch_indices]

                # Convert targets to TensorFlow tensor for GPU processing
                tf_targets = tf.convert_to_tensor(targets, dtype=tf.float32)

                # Train the model using the built-in train_on_batch method on GPU
                loss = self.models[agent_idx].train_on_batch(tf_states, tf_targets)

                # Add the loss to our list
                losses.append(loss)

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Periodically update target networks
            self._update_target_models()

        return np.mean(losses) if losses else None

    def plan_paths(self, grid_state, agent_positions, target_positions, max_steps=100, training=True):
        """
        Plan paths for all agents using the trained models.

        Args:
            grid_state: Dictionary of cell states
            agent_positions: List of agent positions
            target_positions: List of target positions
            max_steps: Maximum number of steps
            training: Whether to update the models during planning

        Returns:
            Dictionary with paths, rewards, and success status for each agent
        """
        if not ML_AVAILABLE or len(agent_positions) != self.num_agents:
            return None

        # Initialize paths and states
        paths = [[] for _ in range(self.num_agents)]
        total_rewards = [0 for _ in range(self.num_agents)]
        dones = [False for _ in range(self.num_agents)]

        # Add initial positions to paths
        for i in range(self.num_agents):
            paths[i].append(agent_positions[i])

        # Main planning loop
        for step in range(max_steps):
            current_positions = [paths[i][-1] for i in range(self.num_agents)]
            next_positions = current_positions.copy()

            # Determine actions for all agents
            for i in range(self.num_agents):
                if dones[i]:
                    continue

                # Get enhanced state representation with all target positions
                other_positions = [pos for j, pos in enumerate(current_positions) if j != i]
                state = self._get_state_representation(grid_state, current_positions[i],
                                                      target_positions[i], other_positions,
                                                      all_target_positions=target_positions)

                # Ensure state has consistent type and shape
                state = np.asarray(state, dtype=np.float32)

                # Get valid actions
                valid_actions = self.get_valid_actions(grid_state, current_positions[i], current_positions)

                if not valid_actions:
                    continue

                # Choose action with consistent state representation
                action_idx = self.act(i, state, valid_actions, training)

                # Apply action
                dr, dc = self.actions[action_idx]
                new_r, new_c = current_positions[i][0] + dr, current_positions[i][1] + dc
                next_positions[i] = (new_r, new_c)

                # Check if target reached
                if next_positions[i] == target_positions[i]:
                    dones[i] = True

            # Check for collisions and resolve them
            for i in range(self.num_agents):
                if dones[i]:
                    continue

                # Check if two agents want to move to the same position
                collision = False
                for j in range(self.num_agents):
                    if i != j and next_positions[i] == next_positions[j]:
                        collision = True
                        break

                # Check if an agent wants to move to another agent's current position
                # while that agent is also moving
                for j in range(self.num_agents):
                    if (i != j and next_positions[i] == current_positions[j] and
                        next_positions[j] != current_positions[j]):
                        collision = True
                        break

                if collision:
                    next_positions[i] = current_positions[i]  # Stay in place

            # Update paths and calculate rewards
            for i in range(self.num_agents):
                if dones[i]:
                    continue

                # Calculate reward with cooperative information
                reward = self.calculate_reward(
                    current_positions[i], next_positions[i],
                    target_positions[i], next_positions[i] == target_positions[i],
                    all_agent_positions=current_positions,
                    all_target_positions=target_positions,
                    all_dones=dones
                )
                total_rewards[i] += reward

                # Add to path
                if next_positions[i] != current_positions[i]:
                    paths[i].append(next_positions[i])

                # Check if target reached
                if next_positions[i] == target_positions[i]:
                    dones[i] = True

                # Store experience if in training mode
                if training:
                    other_positions = [pos for j, pos in enumerate(current_positions) if j != i]
                    state = self._get_state_representation(grid_state, current_positions[i],
                                                          target_positions[i], other_positions,
                                                          all_target_positions=target_positions)

                    # Ensure state has consistent type and shape
                    state = np.asarray(state, dtype=np.float32)

                    other_next_positions = [pos for j, pos in enumerate(next_positions) if j != i]
                    next_state = self._get_state_representation(grid_state, next_positions[i],
                                                               target_positions[i], other_next_positions,
                                                               all_target_positions=target_positions)

                    # Ensure next_state has consistent type and shape
                    next_state = np.asarray(next_state, dtype=np.float32)

                    # Determine which action was taken
                    if next_positions[i] == current_positions[i]:
                        # Agent didn't move (collision)
                        action_idx = -1  # Special value for no movement
                    else:
                        dr = next_positions[i][0] - current_positions[i][0]
                        dc = next_positions[i][1] - current_positions[i][1]
                        try:
                            action_idx = self.actions.index((dr, dc))
                        except ValueError:
                            # If action not found, skip storing this experience
                            action_idx = -1

                    if action_idx >= 0:  # Only remember valid actions
                        # Store experience with consistent types
                        self.remember(i, state, action_idx, float(reward), next_state, bool(dones[i]))

            # Check if all agents reached their targets
            if all(dones):
                break

        # Train the models if in training mode
        if training:
            self.train()

        # Update training metrics
        if training:
            self.training_history['episode_rewards'].append(sum(total_rewards))
            self.training_history['episode_lengths'].append(step + 1)
            self.training_history['success_rate'].append(sum(dones) / self.num_agents if self.num_agents > 0 else 0)

        # Return results
        return {
            'paths': paths,
            'rewards': total_rewards,
            'success': dones,
            'steps': step + 1
        }

    def save_models(self):
        """Save all agent models to disk with version information."""
        if not ML_AVAILABLE:
            return

        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Save version information
        # Current version: 3.1 (enhanced for shape formation and collision avoidance)
        version_info = {
            "version": 3.1,
            "state_size": self.grid_size * self.grid_size + 24,  # Current state size with shape metrics
            "model_architecture": "dueling_dqn_shape_formation_collision_avoidance",
            "timestamp": time.time(),
            "num_agents": self.num_agents,
            "grid_size": self.grid_size,
            "features": [
                "agent_position",
                "target_position",
                "path_info",
                "nearest_agents",
                "shape_metrics",
                "grid_state"
            ],
            "shape_metrics": [
                "shape_error",
                "shape_completion",
                "position_in_formation",
                "is_perimeter",
                "neighbor_count"
            ],
            "collision_avoidance": {
                "enabled": True,
                "collision_penalty": -2.5,
                "safe_distance_reward": 1.0,
                "path_crossing_penalty": -1.5
            }
        }

        version_path = os.path.join(self.model_dir, "version_info.json")
        with open(version_path, "w") as f:
            json.dump(version_info, f)

        # Save each agent's model
        for i in range(self.num_agents):
            model_path = os.path.join(self.model_dir, f"agent_{i}_model.h5")
            self.models[i].save(model_path)

        # Save training history
        history_path = os.path.join(self.model_dir, "training_history.pkl")
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)

        print(f"Saved {self.num_agents} agent models to {self.model_dir} with shape formation version information")

    def load_models(self):
        """Load all agent models from disk with compatibility checking."""
        if not ML_AVAILABLE:
            return False

        # Check if model directory exists
        if not os.path.exists(self.model_dir):
            print(f"Model directory {self.model_dir} does not exist. Using new shape formation models.")
            return False

        # Check version compatibility
        version_path = os.path.join(self.model_dir, "version_info.json")
        if os.path.exists(version_path):
            try:
                with open(version_path, "r") as f:
                    version_info = json.load(f)

                current_state_size = self.grid_size * self.grid_size + 24  # Current expected state size with shape metrics
                saved_state_size = version_info.get("state_size", 0)
                saved_version = version_info.get("version", 0)
                saved_grid_size = version_info.get("grid_size", 0)
                saved_architecture = version_info.get("model_architecture", "")

                # Check for compatibility issues
                if (saved_version < 3.1 or
                    saved_state_size != current_state_size or
                    saved_grid_size != self.grid_size or
                    "collision_avoidance" not in saved_architecture):

                    print(f"Incompatible model version detected (saved: {saved_version}, current: 3.1)")
                    print(f"State size mismatch: saved={saved_state_size}, current={current_state_size}")
                    print(f"Grid size mismatch: saved={saved_grid_size}, current={self.grid_size}")
                    print(f"Architecture mismatch: saved={saved_architecture}, current=dueling_dqn_shape_formation_collision_avoidance")
                    print("Creating new shape formation models with collision avoidance instead of loading incompatible ones.")

                    # Backup old models
                    backup_folder = f"{self.model_dir}_backup_{int(time.time())}"
                    if not os.path.exists(backup_folder):
                        try:
                            import shutil
                            shutil.copytree(self.model_dir, backup_folder)
                            print(f"Backed up old models to {backup_folder}")
                        except Exception as e:
                            print(f"Failed to backup old models: {str(e)}")

                    return False
            except Exception as e:
                print(f"Error reading version information: {str(e)}")
                print("Using new shape formation models as a precaution.")
                return False
        else:
            print("No version information found. Models may be incompatible.")
            print("Creating new shape formation models to ensure compatibility.")
            return False

        # Load each agent's model
        models_loaded = 0
        load_failed = False

        for i in range(self.num_agents):
            model_path = os.path.join(self.model_dir, f"agent_{i}_model.h5")
            if os.path.exists(model_path):
                try:
                    self.models[i] = load_model(model_path)
                    self.target_models[i].set_weights(self.models[i].get_weights())
                    models_loaded += 1
                except Exception as e:
                    print(f"Error loading model for agent {i}: {e}")
                    load_failed = True
                    break
            else:
                print(f"Model for agent {i} not found.")
                load_failed = True
                break

        # If any model failed to load, reinitialize all models for consistency
        if load_failed:
            print("Some models failed to load. Using new models for all agents.")
            self.models = [self._build_model() for _ in range(self.num_agents)]
            self.target_models = [self._build_model() for _ in range(self.num_agents)]
            self._update_target_models()
            return False

        # Load training history
        history_path = os.path.join(self.model_dir, "training_history.pkl")
        if os.path.exists(history_path):
            try:
                with open(history_path, 'rb') as f:
                    self.training_history = pickle.load(f)
            except Exception as e:
                print(f"Error loading training history: {e}")

        if models_loaded > 0:
            print(f"Successfully loaded {models_loaded} agent models from {self.model_dir}")
            return True

        return False

    def train_agents(self, grid_state, num_episodes=1000, max_steps=200,
                     save_interval=100, print_interval=10, curriculum_learning=True):
        """
        Train agents through multiple episodes with curriculum learning for shape formation.

        The curriculum learning approach gradually increases the complexity of the shapes
        that agents need to form, starting with simple formations and progressing to more
        complex ones as agents improve.

        Args:
            grid_state: Dictionary of cell states
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            save_interval: Interval for saving models
            print_interval: Interval for printing progress
            curriculum_learning: Whether to use curriculum learning approach

        Returns:
            Training history or None if training fails
        """
        if not ML_AVAILABLE:
            print("ML libraries not available. Cannot train agents.")
            return None

        # Initialize training history if it doesn't exist
        if not hasattr(self, 'training_history') or self.training_history is None:
            self.training_history = {
                'episode_rewards': [],
                'episode_lengths': [],
                'success_rate': []
            }

        # Check for GPU availability and configure for optimal performance
        gpu_available = False
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                gpu_available = True
                print(f"Found {len(gpus)} GPU(s): {gpus}")
                print(f"Training will use GPU: {gpus[0].name}")

                # Set memory growth again to ensure it's enabled
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Try to enable mixed precision for faster training
                try:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    print("Mixed precision training enabled (FP16)")
                except Exception as e:
                    print(f"Could not enable mixed precision: {e}")
            else:
                print("No GPU found, training will use CPU (slower)")
        except Exception as e:
            print(f"Error checking GPU: {e}")
            print("Training will proceed with available devices")

        print("="*50)
        print(f"STARTING SHAPE FORMATION TRAINING FOR {num_episodes} EPISODES")
        print(f"Number of agents: {self.num_agents}")
        print(f"Grid size: {self.grid_size}x{self.grid_size}")
        print(f"Maximum steps per episode: {max_steps}")
        print(f"Print interval: Every {print_interval} episodes")
        print(f"Save interval: Every {save_interval} episodes")
        print(f"Curriculum learning: {'Enabled' if curriculum_learning else 'Disabled'}")
        print(f"GPU acceleration: {'Enabled' if gpu_available else 'Disabled'}")
        print("="*50)

        # Print initial status
        print("Episode 0 - Starting with epsilon: {:.4f}".format(self.epsilon))

        # Define shape templates for curriculum learning
        shape_templates = self._generate_shape_templates()

        # Track shape formation metrics
        shape_metrics = {
            'completion_rate': [],
            'formation_error': [],
            'current_curriculum_level': 0
        }

        # Initialize curriculum level
        curriculum_level = 0
        curriculum_progress = 0
        curriculum_threshold = 0.8  # Success rate threshold to advance curriculum

        for episode in range(num_episodes):
            # Determine whether to use curriculum learning for this episode
            use_curriculum = curriculum_learning and episode > 0

            # Generate positions based on curriculum level if enabled
            if use_curriculum:
                # Check if we should advance to the next curriculum level
                if (curriculum_progress >= 5 and  # Need at least 5 episodes at current level
                    len(self.training_history['success_rate']) >= 5 and
                    sum(self.training_history['success_rate'][-5:]) / 5 >= curriculum_threshold):

                    # Advance curriculum level if not at max
                    if curriculum_level < len(shape_templates) - 1:
                        curriculum_level += 1
                        curriculum_progress = 0
                        print(f"\n=== ADVANCING TO CURRICULUM LEVEL {curriculum_level + 1} ===")
                        print(f"Shape complexity: {shape_templates[curriculum_level]['name']}")

                # Track current curriculum level
                shape_metrics['current_curriculum_level'] = curriculum_level
                curriculum_progress += 1

                # Get agent and target positions from the current curriculum level
                agent_positions, target_positions = self._generate_curriculum_positions(
                    grid_state,
                    shape_templates[curriculum_level],
                    episode
                )
            else:
                # Generate random positions for non-curriculum episodes
                agent_positions, target_positions = self._generate_random_positions(grid_state)

            # Print agent and target positions for first episode
            if episode == 0:
                print("\nInitial positions for first episode:")
                for i in range(min(self.num_agents, len(agent_positions))):
                    print(f"Agent {i+1}: {agent_positions[i]} → Target: {target_positions[i]}")
                print()

            # Plan paths for this episode
            result = self.plan_paths(grid_state, agent_positions, target_positions,
                                    max_steps=max_steps, training=True)

            # Calculate shape formation metrics
            if result is not None:
                # Calculate shape error at the end of the episode
                final_positions = []
                for i, path in enumerate(result['paths']):
                    if result['success'][i] and path:
                        final_positions.append(path[-1])
                    elif i < len(agent_positions):
                        final_positions.append(agent_positions[i])

                # Calculate shape error if we have positions
                if final_positions:
                    shape_error = self._calculate_shape_error(final_positions, target_positions)
                    shape_metrics['formation_error'].append(shape_error)

                    # Calculate completion rate (percentage of targets reached)
                    completion_rate = sum(result['success']) / len(target_positions) if target_positions else 0
                    shape_metrics['completion_rate'].append(completion_rate)
            else:
                # If result is None, record failure metrics
                shape_metrics['formation_error'].append(1.0)  # Maximum error
                shape_metrics['completion_rate'].append(0.0)  # No completion

            # Print progress for every episode (more verbose)
            if (episode + 1) % print_interval == 0 or episode == 0:
                # Check if result is None
                if result is None:
                    print(f"Episode {episode + 1}/{num_episodes} - "
                          f"Result is None (path planning failed), "
                          f"Epsilon: {self.epsilon:.4f}")
                    continue

                success_rate = sum(result['success']) / self.num_agents
                avg_reward = sum(result['rewards']) / self.num_agents

                # Calculate average shape error for recent episodes
                recent_errors = shape_metrics['formation_error'][-print_interval:] if shape_metrics['formation_error'] else [0]
                avg_shape_error = sum(recent_errors) / len(recent_errors)

                # Calculate average completion rate for recent episodes
                recent_completions = shape_metrics['completion_rate'][-print_interval:] if shape_metrics['completion_rate'] else [0]
                avg_completion = sum(recent_completions) / len(recent_completions)

                print(f"Episode {episode + 1}/{num_episodes} - "
                      f"Success rate: {success_rate:.2f}, "
                      f"Avg reward: {avg_reward:.2f}, "
                      f"Steps: {result['steps']}, "
                      f"Epsilon: {self.epsilon:.4f}")

                if use_curriculum:
                    print(f"  Curriculum level: {curriculum_level + 1}/{len(shape_templates)}, "
                          f"Shape: {shape_templates[curriculum_level]['name']}, "
                          f"Avg completion: {avg_completion:.2f}, "
                          f"Avg shape error: {avg_shape_error:.2f}")

                # Print individual agent results occasionally
                if (episode + 1) % (print_interval * 5) == 0 and result is not None:
                    print("  Agent details:")
                    for i in range(min(self.num_agents, len(agent_positions))):
                        print(f"  - Agent {i+1}: {'Success' if i < len(result['success']) and result['success'][i] else 'Failed'}, "
                              f"Reward: {result['rewards'][i] if i < len(result['rewards']) else 0:.2f}, "
                              f"Path length: {len(result['paths'][i]) if i < len(result['paths']) else 0}")

            # Save models periodically
            if (episode + 1) % save_interval == 0:
                self.save_models()
                print(f"Models saved at episode {episode + 1}")

        # Final save
        self.save_models()

        print("\n" + "="*50)
        print("SHAPE FORMATION TRAINING COMPLETED")
        print(f"Final epsilon value: {self.epsilon:.4f}")
        print(f"Final curriculum level: {curriculum_level + 1}/{len(shape_templates)}")
        print(f"Models saved to: {self.model_dir}")
        print("="*50)

        # Add shape metrics to training history
        self.training_history['shape_metrics'] = shape_metrics

        return self.training_history

    def _generate_random_positions(self, grid_state):
        """
        Generate random agent and target positions.

        Args:
            grid_state: Dictionary of cell states

        Returns:
            Tuple of (agent_positions, target_positions)
        """
        agent_positions = []
        target_positions = []

        # Generate non-overlapping positions
        all_positions = set()

        for _ in range(self.num_agents):
            # Find a free position for the agent
            while True:
                r = random.randint(0, self.grid_size - 1)
                c = random.randint(0, self.grid_size - 1)
                pos = (r, c)

                if (pos not in all_positions and
                    not grid_state.get(pos, {}).get("obstacle", False)):
                    agent_positions.append(pos)
                    all_positions.add(pos)
                    break

            # Find a free position for the target
            while True:
                r = random.randint(0, self.grid_size - 1)
                c = random.randint(0, self.grid_size - 1)
                pos = (r, c)

                if (pos not in all_positions and
                    not grid_state.get(pos, {}).get("obstacle", False)):
                    target_positions.append(pos)
                    all_positions.add(pos)
                    break

        return agent_positions, target_positions

    def _generate_curriculum_positions(self, grid_state, shape_template, episode):
        """
        Generate positions based on curriculum learning level.

        Args:
            grid_state: Dictionary of cell states
            shape_template: Template for the shape to form
            episode: Current episode number

        Returns:
            Tuple of (agent_positions, target_positions)
        """
        # Get shape type and parameters
        shape_type = shape_template['type']

        # Generate target positions based on shape type
        if shape_type == 'line':
            target_positions = self._generate_line_shape(shape_template)
        elif shape_type == 'rectangle':
            target_positions = self._generate_rectangle_shape(shape_template)
        elif shape_type == 'triangle':
            target_positions = self._generate_triangle_shape(shape_template)
        elif shape_type == 'circle':
            target_positions = self._generate_circle_shape(shape_template)
        else:
            # Fallback to random positions
            return self._generate_random_positions(grid_state)

        # Limit target positions to the number of agents
        target_positions = target_positions[:self.num_agents]

        # Generate agent positions
        agent_positions = []
        all_positions = set(target_positions)  # Avoid overlap with targets

        # Place agents randomly but with increasing difficulty as episodes progress
        # Early episodes: agents start close to targets
        # Later episodes: agents start randomly

        # Calculate difficulty factor (0.0 to 1.0)
        difficulty = min(1.0, episode / 200)  # Gradually increase difficulty over 200 episodes

        for target_pos in target_positions:
            # Determine maximum distance from target based on difficulty
            max_distance = int(self.grid_size * difficulty)

            # Ensure minimum distance for challenge
            min_distance = 2 if difficulty > 0.3 else 1

            # Find a position for the agent
            attempts = 0
            while attempts < 100:  # Limit attempts to avoid infinite loop
                if difficulty < 0.3:
                    # Early episodes: place agents close to their targets
                    dr = random.randint(-max_distance, max_distance)
                    dc = random.randint(-max_distance, max_distance)
                    r = max(0, min(self.grid_size - 1, target_pos[0] + dr))
                    c = max(0, min(self.grid_size - 1, target_pos[1] + dc))
                else:
                    # Later episodes: place agents randomly
                    r = random.randint(0, self.grid_size - 1)
                    c = random.randint(0, self.grid_size - 1)

                pos = (r, c)

                # Check if position is valid
                if (pos not in all_positions and
                    not grid_state.get(pos, {}).get("obstacle", False)):

                    # Calculate distance to target
                    dist = abs(pos[0] - target_pos[0]) + abs(pos[1] - target_pos[1])

                    # Ensure minimum distance for challenge
                    if dist >= min_distance:
                        agent_positions.append(pos)
                        all_positions.add(pos)
                        break

                attempts += 1

            # If we couldn't find a valid position, place agent randomly
            if len(agent_positions) < len(target_positions):
                while True:
                    r = random.randint(0, self.grid_size - 1)
                    c = random.randint(0, self.grid_size - 1)
                    pos = (r, c)

                    if (pos not in all_positions and
                        not grid_state.get(pos, {}).get("obstacle", False)):
                        agent_positions.append(pos)
                        all_positions.add(pos)
                        break

        return agent_positions, target_positions

    def _generate_shape_templates(self):
        """
        Generate templates for different shapes used in curriculum learning.

        Returns:
            List of shape templates in order of increasing complexity
        """
        # Calculate center of the grid
        center_r = self.grid_size // 2
        center_c = self.grid_size // 2

        # Define shape templates in order of increasing complexity
        templates = [
            # Level 1: Simple line (horizontal)
            {
                'name': 'Horizontal Line',
                'type': 'line',
                'start': (center_r, center_c - min(2, center_c)),
                'end': (center_r, center_c + min(2, self.grid_size - center_c - 1)),
                'orientation': 'horizontal'
            },

            # Level 2: Simple line (vertical)
            {
                'name': 'Vertical Line',
                'type': 'line',
                'start': (center_r - min(2, center_r), center_c),
                'end': (center_r + min(2, self.grid_size - center_r - 1), center_c),
                'orientation': 'vertical'
            },

            # Level 3: Small rectangle
            {
                'name': 'Small Rectangle',
                'type': 'rectangle',
                'top_left': (center_r - 1, center_c - 1),
                'bottom_right': (center_r + 1, center_c + 1),
                'filled': False
            },

            # Level 4: Small triangle
            {
                'name': 'Triangle',
                'type': 'triangle',
                'top': (center_r - 2, center_c),
                'bottom_left': (center_r + 1, center_c - 2),
                'bottom_right': (center_r + 1, center_c + 2)
            },

            # Level 5: Larger rectangle
            {
                'name': 'Medium Rectangle',
                'type': 'rectangle',
                'top_left': (center_r - 2, center_c - 3),
                'bottom_right': (center_r + 2, center_c + 3),
                'filled': False
            },

            # Level 6: Small circle
            {
                'name': 'Small Circle',
                'type': 'circle',
                'center': (center_r, center_c),
                'radius': 2
            },

            # Level 7: Filled rectangle
            {
                'name': 'Filled Rectangle',
                'type': 'rectangle',
                'top_left': (center_r - 2, center_c - 2),
                'bottom_right': (center_r + 2, center_c + 2),
                'filled': True
            },

            # Level 8: Larger circle
            {
                'name': 'Medium Circle',
                'type': 'circle',
                'center': (center_r, center_c),
                'radius': 3
            }
        ]

        return templates

    def _generate_line_shape(self, template):
        """Generate positions for a line shape."""
        positions = []

        start = template['start']
        end = template['end']
        orientation = template['orientation']

        if orientation == 'horizontal':
            r = start[0]
            for c in range(start[1], end[1] + 1):
                positions.append((r, c))
        else:  # vertical
            c = start[1]
            for r in range(start[0], end[0] + 1):
                positions.append((r, c))

        return positions

    def _generate_rectangle_shape(self, template):
        """Generate positions for a rectangle shape."""
        positions = []

        top_left = template['top_left']
        bottom_right = template['bottom_right']
        filled = template['filled']

        if filled:
            # Generate all positions within the rectangle
            for r in range(top_left[0], bottom_right[0] + 1):
                for c in range(top_left[1], bottom_right[1] + 1):
                    positions.append((r, c))
        else:
            # Generate only the perimeter
            for r in range(top_left[0], bottom_right[0] + 1):
                for c in range(top_left[1], bottom_right[1] + 1):
                    if (r == top_left[0] or r == bottom_right[0] or
                        c == top_left[1] or c == bottom_right[1]):
                        positions.append((r, c))

        return positions

    def _generate_triangle_shape(self, template):
        """Generate positions for a triangle shape."""
        positions = []

        top = template['top']
        bottom_left = template['bottom_left']
        bottom_right = template['bottom_right']

        # Add the three corners
        positions.append(top)
        positions.append(bottom_left)
        positions.append(bottom_right)

        # Add points along the edges
        # Top to bottom left
        r_diff = bottom_left[0] - top[0]
        c_diff = bottom_left[1] - top[1]
        steps = max(abs(r_diff), abs(c_diff))

        if steps > 1:
            for i in range(1, steps):
                r = top[0] + int(r_diff * i / steps)
                c = top[1] + int(c_diff * i / steps)
                positions.append((r, c))

        # Top to bottom right
        r_diff = bottom_right[0] - top[0]
        c_diff = bottom_right[1] - top[1]
        steps = max(abs(r_diff), abs(c_diff))

        if steps > 1:
            for i in range(1, steps):
                r = top[0] + int(r_diff * i / steps)
                c = top[1] + int(c_diff * i / steps)
                positions.append((r, c))

        # Bottom left to bottom right
        r_diff = bottom_right[0] - bottom_left[0]
        c_diff = bottom_right[1] - bottom_left[1]
        steps = max(abs(r_diff), abs(c_diff))

        if steps > 1:
            for i in range(1, steps):
                r = bottom_left[0] + int(r_diff * i / steps)
                c = bottom_left[1] + int(c_diff * i / steps)
                positions.append((r, c))

        return positions

    def _generate_circle_shape(self, template):
        """Generate positions for a circle shape."""
        positions = []

        center = template['center']
        radius = template['radius']

        # Generate points around the circle
        for r in range(center[0] - radius, center[0] + radius + 1):
            for c in range(center[1] - radius, center[1] + radius + 1):
                # Check if point is within grid bounds
                if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                    # Calculate distance from center
                    dist = ((r - center[0]) ** 2 + (c - center[1]) ** 2) ** 0.5

                    # Add points on the circle perimeter (with some tolerance)
                    if abs(dist - radius) < 0.8:
                        positions.append((r, c))

        return positions

    def visualize_paths(self, grid_state, paths, agent_positions, target_positions):
        """
        Generate a text-based visualization of the paths.

        Args:
            grid_state: Dictionary of cell states
            paths: List of paths for each agent
            agent_positions: Initial positions of agents
            target_positions: Target positions for agents

        Returns:
            String representation of the grid with paths
        """
        # Create a grid filled with empty spaces
        grid = [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Mark obstacles
        for pos, cell in grid_state.items():
            if cell.get("obstacle", False):
                grid[pos[0]][pos[1]] = '█'

        # Mark paths with numbers
        for i, path in enumerate(paths):
            for step, pos in enumerate(path):
                # Skip start and end positions
                if step > 0 and step < len(path) - 1:
                    grid[pos[0]][pos[1]] = str(i + 1)

        # Mark start positions with S
        for i, pos in enumerate(agent_positions):
            grid[pos[0]][pos[1]] = f'S{i+1}'

        # Mark target positions with T
        for i, pos in enumerate(target_positions):
            grid[pos[0]][pos[1]] = f'T{i+1}'

        # Convert grid to string
        grid_str = '+' + '-' * (self.grid_size * 2 - 1) + '+\n'
        for row in grid:
            grid_str += '|' + '|'.join(row) + '|\n'
        grid_str += '+' + '-' * (self.grid_size * 2 - 1) + '+'

        return grid_str


# Example usage
if __name__ == "__main__":
    # Create a sample grid with obstacles
    grid_size = 10
    grid_state = {}

    # Add some obstacles
    for i in range(3, 7):
        grid_state[(i, 4)] = {"obstacle": True}

    for i in range(2, 5):
        grid_state[(7, i)] = {"obstacle": True}

    # Initialize MARL system
    marl = PathPlanningMARL(grid_size=grid_size, num_agents=3)

    # Set initial positions and targets
    agent_positions = [(0, 0), (0, 9), (9, 0)]
    target_positions = [(9, 9), (9, 5), (5, 9)]

    # Plan paths
    result = marl.plan_paths(grid_state, agent_positions, target_positions, max_steps=50, training=False)

    if result:
        # Visualize the paths
        print(marl.visualize_paths(grid_state, result['paths'], agent_positions, target_positions))

        # Print results
        for i in range(len(agent_positions)):
            print(f"Agent {i+1} - Success: {result['success'][i]}, Reward: {result['rewards'][i]:.2f}")
            print(f"Path: {result['paths'][i]}")
            print()

    # Train the agents
    # Uncomment to run training
    # marl.train_agents(grid_state, num_episodes=100, max_steps=50)
