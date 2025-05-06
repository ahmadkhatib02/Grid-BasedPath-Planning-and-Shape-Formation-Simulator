"""
Integration module for adding MARL capabilities to the Interactive Grid application.
This module focuses on training agents for parallel decentralized movement.
"""

import threading
import time
import os
import tkinter as tk
from tkinter import ttk, Frame, scrolledtext
import numpy as np

# Import the MARL module
from PathPlanningMARL import PathPlanningMARL

class MARLIntegration:
    """
    Class to integrate MARL with the Interactive Grid application.
    Specifically designed to improve parallel decentralized movement.
    """

    def __init__(self, parent_app):
        """
        Initialize the MARL integration.

        Args:
            parent_app: The parent InteractiveGrid application
        """
        self.parent = parent_app
        self.marl = None
        self.training_thread = None
        self.training_active = False
        self.training_progress = 0
        self.training_episodes = 100

        # Create UI elements for MARL
        self.create_marl_ui()

        # Training metrics
        self.metrics = {
            'episode_rewards': [],
            'success_rates': [],
            'episode_lengths': [],
            'epsilon_values': []
        }

    def create_marl_ui(self):
        """Create UI elements for MARL integration."""
        # Create a new frame for MARL controls
        marl_header = Frame(self.parent.root, bg="#34495E", padx=5, pady=2)
        marl_header.pack(fill=tk.X, pady=(10, 0))

        tk.Label(marl_header, text="Multi-Agent Reinforcement Learning",
                font=("Arial", 10, "bold"), bg="#34495E", fg="white").pack(anchor=tk.W)

        marl_frame = Frame(self.parent.root, bg="#F8F9FA")
        marl_frame.pack(pady=5, fill=tk.X)

        # MARL settings frame
        settings_frame = Frame(marl_frame, bg="#F8F9FA")
        settings_frame.pack(pady=5, fill=tk.X)

        # Episodes setting
        episodes_frame = Frame(settings_frame, bg="#F8F9FA")
        episodes_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(episodes_frame, text="Episodes:", bg="#F8F9FA").pack(side=tk.LEFT)
        self.episodes_var = tk.IntVar(value=300)  # Increased default episodes
        episodes_entry = tk.Entry(episodes_frame, textvariable=self.episodes_var, width=5)
        episodes_entry.pack(side=tk.LEFT, padx=5)

        # Curriculum learning option
        curriculum_frame = Frame(settings_frame, bg="#F8F9FA")
        curriculum_frame.pack(side=tk.LEFT, padx=10)

        self.curriculum_var = tk.BooleanVar(value=True)
        curriculum_check = tk.Checkbutton(curriculum_frame, text="Curriculum Learning",
                                         variable=self.curriculum_var, bg="#F8F9FA")
        curriculum_check.pack(side=tk.LEFT)

        # Staged training option
        staged_frame = Frame(settings_frame, bg="#F8F9FA")
        staged_frame.pack(side=tk.LEFT, padx=10)

        self.staged_var = tk.BooleanVar(value=True)
        staged_check = tk.Checkbutton(staged_frame, text="Staged Training",
                                     variable=self.staged_var, bg="#F8F9FA")
        staged_check.pack(side=tk.LEFT)

        # MARL buttons frame
        buttons_frame = Frame(marl_frame, bg="#F8F9FA")
        buttons_frame.pack(pady=5, fill=tk.X)

        # Button style
        button_style = {
            "bg": "#3498DB",
            "fg": "white",
            "activebackground": "#2980B9",
            "relief": tk.RAISED,
            "padx": 10,
            "pady": 5
        }

        # Train button
        self.train_btn = tk.Button(
            buttons_frame,
            text="Train MARL for Parallel Movement",
            command=self.start_marl_training,
            **button_style
        )
        self.train_btn.pack(side=tk.LEFT, padx=5)

        # Path blocking training button
        path_blocking_style = button_style.copy()
        path_blocking_style["bg"] = "#FF9800"
        path_blocking_style["activebackground"] = "#F57C00"

        self.path_blocking_btn = tk.Button(
            buttons_frame,
            text="Train for Path Blocking",
            command=self.train_for_path_blocking,
            **path_blocking_style
        )
        self.path_blocking_btn.pack(side=tk.LEFT, padx=5)

        # Use MARL button
        self.use_marl_btn = tk.Button(
            buttons_frame,
            text="Do Shape with MARL",
            command=self.do_shape_with_marl,
            state=tk.DISABLED,
            **button_style
        )
        self.use_marl_btn.pack(side=tk.LEFT, padx=5)

        # Progress bar
        progress_frame = Frame(marl_frame, bg="#F8F9FA")
        progress_frame.pack(pady=5, fill=tk.X)

        tk.Label(progress_frame, text="Training Progress:", bg="#F8F9FA").pack(side=tk.LEFT)
        self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_label = tk.Label(progress_frame, text="Not started", bg="#F8F9FA")
        self.status_label.pack(side=tk.LEFT, padx=5)

    def start_marl_training(self):
        """Start MARL training for parallel movement."""
        if self.training_active:
            self.parent.update_status("Training already in progress!")
            return

        # Get training episodes
        try:
            self.training_episodes = self.episodes_var.get()
            if self.training_episodes < 10:
                self.parent.update_status("Episodes must be at least 10")
                return
        except:
            self.parent.update_status("Invalid episode count")
            return

        # Convert grid to MARL format
        grid_state = self.convert_grid_to_marl_format()

        # Initialize MARL with enhanced parameters for shape formation
        self.marl = PathPlanningMARL(
            grid_size=self.parent.grid_size,
            num_agents=len(self.parent.active_cells),
            epsilon=1.0,  # Start with full exploration
            epsilon_decay=0.997,  # Slower decay for better exploration with more episodes
            epsilon_min=0.05,  # Higher minimum exploration rate
            gamma=0.98,  # Higher discount factor for better long-term planning
            learning_rate=0.0005,  # Lower learning rate for more stable learning
            memory_size=20000,  # Larger memory for better experience diversity
            batch_size=128  # Larger batch size for more stable updates
        )

        # Clear any existing models to ensure compatibility with the new architecture
        import shutil
        import os

        model_dir = self.marl.model_dir
        if os.path.exists(model_dir):
            # Backup old models
            backup_dir = f"{model_dir}_backup_{int(time.time())}"
            try:
                shutil.copytree(model_dir, backup_dir)
                print(f"Backed up old models to {backup_dir}")
                # Remove old models
                shutil.rmtree(model_dir)
                os.makedirs(model_dir)
                print(f"Removed old models and created new model directory")
            except Exception as e:
                print(f"Error handling model directory: {str(e)}")
                # Just create a new directory if backup fails
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)

        # Reset metrics
        self.metrics = {
            'episode_rewards': [],
            'success_rates': [],
            'episode_lengths': [],
            'epsilon_values': []
        }

        # Reset progress
        self.training_progress = 0
        self.progress_bar['value'] = 0
        self.status_label.config(text="Starting...")

        # Disable buttons during training
        self.train_btn.config(state=tk.DISABLED)
        self.use_marl_btn.config(state=tk.DISABLED)
        self.parent.do_shape_btn.config(state=tk.DISABLED)

        # Start training in a separate thread
        self.training_active = True
        self.training_thread = threading.Thread(target=self.run_marl_training, args=(grid_state,))
        self.training_thread.daemon = True
        self.training_thread.start()

        # Start progress update
        self.update_training_progress()

        self.parent.update_status("MARL training started for parallel movement...")

    def run_marl_training(self, grid_state):
        """
        Run MARL training in a separate thread.

        Args:
            grid_state: Dictionary of cell states
        """
        try:
            # Create a callback to update progress
            def progress_callback(episode, total_episodes, metrics):
                self.training_progress = (episode + 1) / total_episodes * 100

                # Store metrics
                if metrics:
                    self.metrics['episode_rewards'].append(metrics.get('reward', 0))
                    self.metrics['success_rates'].append(metrics.get('success_rate', 0))
                    self.metrics['episode_lengths'].append(metrics.get('steps', 0))
                    self.metrics['epsilon_values'].append(metrics.get('epsilon', 0))

            # Custom training loop to get more control and progress updates
            if not hasattr(self.marl, 'ML_AVAILABLE') or not self.marl.ML_AVAILABLE:
                self.parent.root.after(0, lambda: self.parent.update_status(
                    "ML libraries not available. Cannot train agents."))
                self.training_completed(success=False)
                return

            print(f"Starting training for {self.training_episodes} episodes...")

            for episode in range(self.training_episodes):
                if not self.training_active:
                    break

                # Implement curriculum learning and staged training
                use_curriculum = self.curriculum_var.get()
                use_staged = self.staged_var.get()

                # Calculate curriculum difficulty based on episode number
                # Start with easy scenarios and gradually increase difficulty
                if use_curriculum:
                    # Calculate difficulty from 0.0 to 1.0 based on episode progress
                    difficulty = min(1.0, episode / (self.training_episodes * 0.7))
                    max_distance = int(self.marl.grid_size * difficulty) + 2  # Minimum distance of 2

                    # Print curriculum progress occasionally
                    if episode % 20 == 0:
                        print(f"Curriculum learning: difficulty={difficulty:.2f}, max_distance={max_distance}")
                else:
                    # No curriculum - full difficulty
                    max_distance = self.marl.grid_size * 2

                # For staged training, we focus on different aspects in different stages
                if use_staged:
                    # Stage 1 (0-30%): Focus on individual navigation
                    # Stage 2 (30-60%): Focus on avoiding collisions
                    # Stage 3 (60-100%): Focus on full shape formation
                    stage_progress = episode / self.training_episodes

                    if stage_progress < 0.3:
                        # Stage 1: Spread out targets to focus on navigation
                        stage = 1
                        spread_factor = 0.8  # High spread
                        collision_focus = False
                    elif stage_progress < 0.6:
                        # Stage 2: Medium spread, introduce collision avoidance
                        stage = 2
                        spread_factor = 0.5  # Medium spread
                        collision_focus = True
                    else:
                        # Stage 3: Low spread, focus on formation
                        stage = 3
                        spread_factor = 0.3  # Low spread
                        collision_focus = True

                    # Print stage transition
                    if episode % 20 == 0 or (episode > 0 and (episode - 1) / self.training_episodes < 0.3 <= stage_progress) or \
                       (episode > 0 and (episode - 1) / self.training_episodes < 0.6 <= stage_progress):
                        print(f"Staged training: now in Stage {stage}")
                else:
                    # No staged training - balanced approach
                    spread_factor = 0.5
                    collision_focus = True

                # Generate positions based on curriculum and stage settings
                agent_positions = []
                target_positions = []
                all_positions = set()

                # Generate non-overlapping positions with curriculum constraints
                for agent_idx in range(self.marl.num_agents):
                    # Find a free position for the agent
                    while True:
                        r = np.random.randint(0, self.marl.grid_size)
                        c = np.random.randint(0, self.marl.grid_size)
                        pos = (r, c)

                        if (pos not in all_positions and
                            not grid_state.get(pos, {}).get("obstacle", False)):
                            agent_positions.append(pos)
                            all_positions.add(pos)
                            break

                    # Find a free position for the target based on curriculum difficulty
                    attempts = 0
                    while attempts < 100:  # Prevent infinite loops
                        attempts += 1

                        if use_curriculum:
                            # In curriculum mode, control the distance based on difficulty
                            if use_staged and stage == 1:
                                # Stage 1: Random positions for individual navigation practice
                                r = np.random.randint(0, self.marl.grid_size)
                                c = np.random.randint(0, self.marl.grid_size)
                            else:
                                # Generate target within the max_distance from agent
                                # but not too close either (min distance of 2)
                                agent_r, agent_c = agent_positions[agent_idx]

                                # Calculate distance range based on difficulty
                                min_dist = 2
                                curr_max_dist = max(min_dist + 1, max_distance)

                                # Random distance within range
                                dist = np.random.randint(min_dist, curr_max_dist)

                                # Random direction
                                angle = np.random.random() * 2 * np.pi

                                # Calculate position (with bounds checking)
                                r = max(0, min(self.marl.grid_size - 1, int(agent_r + dist * np.cos(angle))))
                                c = max(0, min(self.marl.grid_size - 1, int(agent_c + dist * np.sin(angle))))
                        else:
                            # Without curriculum, just random positions
                            r = np.random.randint(0, self.marl.grid_size)
                            c = np.random.randint(0, self.marl.grid_size)

                        pos = (r, c)

                        # Check if position is valid
                        if (pos not in all_positions and
                            not grid_state.get(pos, {}).get("obstacle", False)):

                            # In staged training with collision focus, ensure targets are
                            # somewhat clustered to practice collision avoidance
                            if collision_focus and agent_idx > 0 and target_positions:
                                # Calculate average distance to other targets
                                avg_dist = sum(abs(r - t[0]) + abs(c - t[1]) for t in target_positions) / len(target_positions)

                                # Accept position based on spread factor
                                # Lower spread factor = more clustering
                                if avg_dist <= self.marl.grid_size * spread_factor:
                                    target_positions.append(pos)
                                    all_positions.add(pos)
                                    break
                                # If not accepted, try again unless we've tried too many times
                                elif attempts > 80:  # Accept anyway after many attempts
                                    target_positions.append(pos)
                                    all_positions.add(pos)
                                    break
                            else:
                                # Without collision focus, accept any valid position
                                target_positions.append(pos)
                                all_positions.add(pos)
                                break

                # Plan paths for this episode
                result = self.marl.plan_paths(
                    grid_state,
                    agent_positions,
                    target_positions,
                    max_steps=100,
                    training=True
                )

                # Update progress
                if result:
                    success_rate = sum(result['success']) / self.marl.num_agents
                    avg_reward = sum(result['rewards']) / self.marl.num_agents

                    metrics = {
                        'reward': avg_reward,
                        'success_rate': success_rate,
                        'steps': result['steps'],
                        'epsilon': self.marl.epsilon
                    }

                    progress_callback(episode, self.training_episodes, metrics)

                    # Print progress occasionally
                    if (episode + 1) % 10 == 0 or episode == 0:
                        print(f"Episode {episode + 1}/{self.training_episodes} - "
                              f"Success rate: {success_rate:.2f}, "
                              f"Avg reward: {avg_reward:.2f}, "
                              f"Steps: {result['steps']}, "
                              f"Epsilon: {self.marl.epsilon:.4f}")

                # Save models periodically
                if (episode + 1) % 20 == 0:
                    self.marl.save_models()

            # Final save
            self.marl.save_models()
            print("Training completed.")

            # Signal completion
            self.training_completed(success=True)

        except Exception as e:
            print(f"Error during training: {str(e)}")
            self.parent.root.after(0, lambda: self.parent.update_status(
                f"Error during MARL training: {str(e)}"))
            self.training_completed(success=False)

    def update_training_progress(self):
        """Update the training progress bar and status."""
        if self.training_active:
            # Update progress bar
            self.progress_bar['value'] = self.training_progress

            # Update status label
            if self.training_progress < 100:
                self.status_label.config(text=f"{self.training_progress:.1f}% complete")
            else:
                self.status_label.config(text="Finalizing...")

            # Schedule next update
            self.parent.root.after(100, self.update_training_progress)

    def training_completed(self, success=True):
        """
        Called when training is completed.

        Args:
            success: Whether training completed successfully
        """
        self.training_active = False

        # Update UI in main thread
        self.parent.root.after(0, lambda: self._update_ui_after_training(success))

    def _update_ui_after_training(self, success):
        """Update UI elements after training completes."""
        # Enable buttons
        self.train_btn.config(state=tk.NORMAL)
        self.path_blocking_btn.config(state=tk.NORMAL)
        self.parent.do_shape_btn.config(state=tk.NORMAL)

        if success:
            # Enable MARL button
            self.use_marl_btn.config(state=tk.NORMAL)

            # Update progress and status
            self.progress_bar['value'] = 100
            self.status_label.config(text="Training complete")

            # Update parent status
            self.parent.update_status("MARL training completed successfully!")

            # Add training results to metrics display
            self.update_parent_metrics()
        else:
            # Update status
            self.progress_bar['value'] = 0
            self.status_label.config(text="Training failed")

            # Update parent status
            self.parent.update_status("MARL training failed. See console for details.")

    def train_for_path_blocking(self):
        """
        Specialized training focused on path blocking scenarios.
        This creates challenging scenarios where agents must navigate around each other.
        """
        if self.training_active:
            self.parent.update_status("Training already in progress!")
            return

        # Get training episodes
        try:
            self.training_episodes = self.episodes_var.get()
            if self.training_episodes < 10:
                self.parent.update_status("Episodes must be at least 10")
                return
        except:
            self.parent.update_status("Invalid episode count")
            return

        # Convert grid to MARL format
        grid_state = self.convert_grid_to_marl_format()

        # Initialize MARL with parameters optimized for path blocking
        self.marl = PathPlanningMARL(
            grid_size=self.parent.grid_size,
            num_agents=len(self.parent.active_cells),
            epsilon=1.0,
            epsilon_decay=0.995,  # Slightly faster decay for more exploitation
            epsilon_min=0.05,
            gamma=0.99,  # Higher discount factor for better long-term planning
            learning_rate=0.0003,  # Lower learning rate for more stable learning
            memory_size=25000,  # Larger memory for better experience diversity
            batch_size=128
        )

        # Clear any existing models to ensure compatibility with the new architecture
        import shutil
        import os

        model_dir = self.marl.model_dir
        if os.path.exists(model_dir):
            # Backup old models
            backup_dir = f"{model_dir}_backup_pathblocking_{int(time.time())}"
            try:
                shutil.copytree(model_dir, backup_dir)
                print(f"Backed up old models to {backup_dir}")
                # Remove old models
                shutil.rmtree(model_dir)
                os.makedirs(model_dir)
                print(f"Removed old models and created new model directory")
            except Exception as e:
                print(f"Error handling model directory: {str(e)}")
                # Just create a new directory if backup fails
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)

        # Reset metrics
        self.metrics = {
            'episode_rewards': [],
            'success_rates': [],
            'episode_lengths': [],
            'epsilon_values': []
        }

        # Reset progress
        self.training_progress = 0
        self.progress_bar['value'] = 0
        self.status_label.config(text="Starting path blocking training...")

        # Disable buttons during training
        self.train_btn.config(state=tk.DISABLED)
        self.path_blocking_btn.config(state=tk.DISABLED)
        self.use_marl_btn.config(state=tk.DISABLED)
        self.parent.do_shape_btn.config(state=tk.DISABLED)

        # Start training in a separate thread
        self.training_active = True
        self.training_thread = threading.Thread(target=self.run_path_blocking_training, args=(grid_state,))
        self.training_thread.daemon = True
        self.training_thread.start()

        # Start progress update
        self.update_training_progress()

        self.parent.update_status("Specialized path blocking training started...")

    def run_path_blocking_training(self, grid_state):
        """
        Run specialized path blocking training in a separate thread.
        Creates scenarios where agents must navigate through tight spaces and around each other.

        Args:
            grid_state: Dictionary of cell states
        """
        try:
            # Create a callback to update progress
            def progress_callback(episode, total_episodes, metrics):
                self.training_progress = (episode + 1) / total_episodes * 100

                # Store metrics
                if metrics:
                    self.metrics['episode_rewards'].append(metrics.get('reward', 0))
                    self.metrics['success_rates'].append(metrics.get('success_rate', 0))
                    self.metrics['episode_lengths'].append(metrics.get('steps', 0))
                    self.metrics['epsilon_values'].append(metrics.get('epsilon', 0))

            # Check if ML is available
            if not hasattr(self.marl, 'ML_AVAILABLE') or not self.marl.ML_AVAILABLE:
                self.parent.root.after(0, lambda: self.parent.update_status(
                    "ML libraries not available. Cannot train agents."))
                self.training_completed(success=False)
                return

            print(f"Starting path blocking training for {self.training_episodes} episodes...")

            for episode in range(self.training_episodes):
                if not self.training_active:
                    break

                # Generate challenging path blocking scenarios
                agent_positions = []
                target_positions = []
                all_positions = set()

                # Determine scenario type for this episode
                # 0: Corridor scenario - agents must navigate through narrow passages
                # 1: Crossing paths scenario - agents must cross each other's paths
                # 2: Bottleneck scenario - multiple agents must go through a tight space
                scenario_type = episode % 3

                # Create base positions for agents based on scenario
                if scenario_type == 0:  # Corridor scenario
                    print(f"Episode {episode+1}: Corridor scenario")
                    self._create_corridor_scenario(grid_state, agent_positions, target_positions, all_positions)
                elif scenario_type == 1:  # Crossing paths
                    print(f"Episode {episode+1}: Crossing paths scenario")
                    self._create_crossing_paths_scenario(grid_state, agent_positions, target_positions, all_positions)
                else:  # Bottleneck
                    print(f"Episode {episode+1}: Bottleneck scenario")
                    self._create_bottleneck_scenario(grid_state, agent_positions, target_positions, all_positions)

                # Plan paths for this episode
                result = self.marl.plan_paths(
                    grid_state,
                    agent_positions,
                    target_positions,
                    max_steps=150,  # Longer steps for complex scenarios
                    training=True
                )

                # Update progress
                if result:
                    success_rate = sum(result['success']) / self.marl.num_agents
                    avg_reward = sum(result['rewards']) / self.marl.num_agents

                    metrics = {
                        'reward': avg_reward,
                        'success_rate': success_rate,
                        'steps': result['steps'],
                        'epsilon': self.marl.epsilon
                    }

                    progress_callback(episode, self.training_episodes, metrics)

                    # Print progress occasionally
                    if (episode + 1) % 5 == 0 or episode == 0:
                        print(f"Episode {episode + 1}/{self.training_episodes} - "
                              f"Success rate: {success_rate:.2f}, "
                              f"Avg reward: {avg_reward:.2f}, "
                              f"Steps: {result['steps']}, "
                              f"Epsilon: {self.marl.epsilon:.4f}")

                # Save models periodically
                if (episode + 1) % 20 == 0:
                    self.marl.save_models()

            # Final save
            self.marl.save_models()
            print("Path blocking training completed.")

            # Signal completion
            self.training_completed(success=True)

        except Exception as e:
            print(f"Error during path blocking training: {str(e)}")
            self.parent.root.after(0, lambda: self.parent.update_status(
                f"Error during path blocking training: {str(e)}"))
            self.training_completed(success=False)

    def _create_corridor_scenario(self, grid_state, agent_positions, target_positions, all_positions):
        """Create a corridor scenario where agents must navigate through narrow passages."""
        grid_size = self.marl.grid_size
        num_agents = min(4, self.marl.num_agents)  # Limit to 4 agents for this scenario

        # Create a corridor in the middle of the grid
        corridor_start = grid_size // 4
        corridor_end = grid_size - corridor_start

        # Place agents on one side of the corridor
        for i in range(num_agents):
            # Try to place agent
            for _ in range(100):  # Prevent infinite loops
                row = np.random.randint(corridor_start, corridor_end)
                col = np.random.randint(0, corridor_start - 1)
                pos = (row, col)

                if pos not in all_positions and not grid_state.get(pos, {}).get("obstacle", False):
                    agent_positions.append(pos)
                    all_positions.add(pos)
                    break

        # Place targets on the other side of the corridor
        for i in range(len(agent_positions)):
            # Try to place target
            for _ in range(100):  # Prevent infinite loops
                row = np.random.randint(corridor_start, corridor_end)
                col = np.random.randint(corridor_end + 1, grid_size - 1)
                pos = (row, col)

                if pos not in all_positions and not grid_state.get(pos, {}).get("obstacle", False):
                    target_positions.append(pos)
                    all_positions.add(pos)
                    break

    def _create_crossing_paths_scenario(self, grid_state, agent_positions, target_positions, all_positions):
        """Create a scenario where agents must cross each other's paths."""
        grid_size = self.marl.grid_size
        num_agents = min(6, self.marl.num_agents)  # Limit to 6 agents for this scenario

        # Place agents in opposite corners
        corners = [
            (0, 0, grid_size // 3, grid_size // 3),  # Top-left
            (0, grid_size - grid_size // 3, grid_size // 3, grid_size),  # Top-right
            (grid_size - grid_size // 3, 0, grid_size, grid_size // 3),  # Bottom-left
            (grid_size - grid_size // 3, grid_size - grid_size // 3, grid_size, grid_size)  # Bottom-right
        ]

        # Assign agents to corners
        agents_per_corner = max(1, num_agents // 4)
        corner_idx = 0

        for i in range(num_agents):
            corner = corners[corner_idx]

            # Try to place agent in this corner
            for _ in range(100):  # Prevent infinite loops
                row = np.random.randint(corner[0], corner[2])
                col = np.random.randint(corner[1], corner[3])
                pos = (row, col)

                if pos not in all_positions and not grid_state.get(pos, {}).get("obstacle", False):
                    agent_positions.append(pos)
                    all_positions.add(pos)
                    break

            # Move to next corner for next agent
            if (i + 1) % agents_per_corner == 0:
                corner_idx = (corner_idx + 1) % 4

        # Place targets in opposite corners
        for i, agent_pos in enumerate(agent_positions):
            # Determine which corner the agent is in
            for c_idx, corner in enumerate(corners):
                if (corner[0] <= agent_pos[0] < corner[2] and
                    corner[1] <= agent_pos[1] < corner[3]):
                    # Place target in opposite corner
                    opposite_corner = corners[(c_idx + 2) % 4]

                    # Try to place target
                    for _ in range(100):  # Prevent infinite loops
                        row = np.random.randint(opposite_corner[0], opposite_corner[2])
                        col = np.random.randint(opposite_corner[1], opposite_corner[3])
                        pos = (row, col)

                        if pos not in all_positions and not grid_state.get(pos, {}).get("obstacle", False):
                            target_positions.append(pos)
                            all_positions.add(pos)
                            break

                    break

    def _create_bottleneck_scenario(self, grid_state, agent_positions, target_positions, all_positions):
        """Create a bottleneck scenario where multiple agents must go through a tight space."""
        grid_size = self.marl.grid_size
        num_agents = min(5, self.marl.num_agents)  # Limit to 5 agents for this scenario

        # Create a bottleneck in the middle of the grid
        center_row = grid_size // 2
        center_col = grid_size // 2
        bottleneck_size = max(1, grid_size // 20)

        # Place agents on one side of the bottleneck
        side = np.random.choice([0, 1])  # 0: left side, 1: right side

        if side == 0:  # Agents on left, targets on right
            # Place agents on the left side
            for i in range(num_agents):
                # Try to place agent
                for _ in range(100):  # Prevent infinite loops
                    row = np.random.randint(center_row - grid_size // 4, center_row + grid_size // 4)
                    col = np.random.randint(0, center_col - bottleneck_size)
                    pos = (row, col)

                    if pos not in all_positions and not grid_state.get(pos, {}).get("obstacle", False):
                        agent_positions.append(pos)
                        all_positions.add(pos)
                        break

            # Place targets on the right side
            for i in range(len(agent_positions)):
                # Try to place target
                for _ in range(100):  # Prevent infinite loops
                    row = np.random.randint(center_row - grid_size // 4, center_row + grid_size // 4)
                    col = np.random.randint(center_col + bottleneck_size, grid_size - 1)
                    pos = (row, col)

                    if pos not in all_positions and not grid_state.get(pos, {}).get("obstacle", False):
                        target_positions.append(pos)
                        all_positions.add(pos)
                        break
        else:  # Agents on right, targets on left
            # Place agents on the right side
            for i in range(num_agents):
                # Try to place agent
                for _ in range(100):  # Prevent infinite loops
                    row = np.random.randint(center_row - grid_size // 4, center_row + grid_size // 4)
                    col = np.random.randint(center_col + bottleneck_size, grid_size - 1)
                    pos = (row, col)

                    if pos not in all_positions and not grid_state.get(pos, {}).get("obstacle", False):
                        agent_positions.append(pos)
                        all_positions.add(pos)
                        break

            # Place targets on the left side
            for i in range(len(agent_positions)):
                # Try to place target
                for _ in range(100):  # Prevent infinite loops
                    row = np.random.randint(center_row - grid_size // 4, center_row + grid_size // 4)
                    col = np.random.randint(0, center_col - bottleneck_size)
                    pos = (row, col)

                    if pos not in all_positions and not grid_state.get(pos, {}).get("obstacle", False):
                        target_positions.append(pos)
                        all_positions.add(pos)
                        break

    def update_parent_metrics(self):
        """Update the parent's metrics display with MARL results."""
        if not self.metrics['episode_rewards']:
            return

        # Get final metrics
        final_success_rate = self.metrics['success_rates'][-1] * 100
        final_steps = self.metrics['episode_lengths'][-1]
        # Also track final reward for debugging
        avg_reward = self.metrics['episode_rewards'][-1]
        print(f"Training completed - Final success rate: {final_success_rate:.1f}%, Avg reward: {avg_reward:.2f}")

        # Add MARL to parent metrics if not already there
        if "MARL" not in self.parent.metrics:
            self.parent.metrics["MARL"] = {
                "explored": 0,
                "path_length": 0,
                "time": 0,
                "moves": 0,
                "success_rate": 0,
                "completed_targets": 0,
                "total_targets": 0
            }

        # Update MARL metrics
        self.parent.metrics["MARL"]["success_rate"] = final_success_rate
        self.parent.metrics["MARL"]["path_length"] = final_steps
        self.parent.metrics["MARL"]["moves"] = self.training_episodes

        # Update metrics display
        self.parent.update_metrics_display()

    def convert_grid_to_marl_format(self):
        """
        Convert the parent's grid to the format expected by MARL.

        Returns:
            Dictionary of cell states in MARL format
        """
        grid_state = {}

        for cell in self.parent.cells:
            grid_state[cell] = {
                "obstacle": self.parent.cells[cell]["obstacle"],
                "active": self.parent.cells[cell]["active"]
            }

        return grid_state

    def do_shape_with_marl(self):
        """Use the trained MARL system to do the shape."""
        if not self.marl:
            self.parent.update_status("Please train MARL first!")
            return

        if self.parent.movement_started:
            self.parent.update_status("Movement already in progress!")
            return

        # Set movement started flag
        self.parent.movement_started = True

        # Disable buttons during movement
        self.train_btn.config(state=tk.DISABLED)
        self.use_marl_btn.config(state=tk.DISABLED)
        self.parent.do_shape_btn.config(state=tk.DISABLED)

        # Convert grid to MARL format
        grid_state = self.convert_grid_to_marl_format()

        # Get agent positions and targets
        agent_positions = self.parent.active_cells.copy()

        # Use target shape as targets, but only as many as we have agents
        target_positions = self.parent.target_shape[:len(agent_positions)]

        # Start time for metrics
        start_time = time.time()

        # Plan paths using MARL
        self.parent.update_status("Planning paths with MARL...")

        result = self.marl.plan_paths(
            grid_state,
            agent_positions,
            target_positions,
            max_steps=100,
            training=False  # Use trained model, don't train
        )

        if result:
            # Calculate metrics
            elapsed_time = time.time() - start_time
            success_rate = sum(result['success']) / len(agent_positions) * 100
            total_path_length = sum(len(path) for path in result['paths'])

            # Update metrics
            if "MARL" not in self.parent.metrics:
                self.parent.metrics["MARL"] = {
                    "explored": 0,
                    "path_length": 0,
                    "time": 0,
                    "moves": 0,
                    "success_rate": 0,
                    "completed_targets": 0,
                    "total_targets": 0
                }

            self.parent.metrics["MARL"]["time"] += elapsed_time
            self.parent.metrics["MARL"]["path_length"] += total_path_length
            self.parent.metrics["MARL"]["moves"] += 1
            self.parent.metrics["MARL"]["success_rate"] = success_rate
            self.parent.metrics["MARL"]["completed_targets"] = sum(result['success'])
            self.parent.metrics["MARL"]["total_targets"] = len(agent_positions)

            # Update metrics display
            self.parent.update_metrics_display()

            # Start parallel movement with MARL paths
            self.parent.update_status(f"Starting movement with MARL paths. Success rate: {success_rate:.1f}%")

            # Use the paths to move agents in parallel
            self.start_marl_movement(agent_positions, target_positions, result['paths'])
        else:
            self.parent.update_status("MARL path planning failed!")
            self.parent.movement_started = False

            # Re-enable buttons
            self.train_btn.config(state=tk.NORMAL)
            self.use_marl_btn.config(state=tk.NORMAL)
            self.parent.do_shape_btn.config(state=tk.NORMAL)

    def start_marl_movement(self, agent_positions, target_positions, paths):
        """
        Start movement using MARL-generated paths.

        Args:
            agent_positions: List of agent positions
            target_positions: List of target positions
            paths: List of paths for each agent
        """
        # Initialize moving cells
        self.parent.moving_cells = {}

        # Start all movements
        for i, (agent_pos, target_pos, path) in enumerate(zip(agent_positions, target_positions, paths)):
            if not path:
                continue

            # Generate a unique ID for this movement
            move_id = f"marl_{i}"

            # Store the path and current index
            self.parent.moving_cells[move_id] = (path, 1, target_pos, agent_pos)

            # Remove cell from active cells
            if agent_pos in self.parent.active_cells:
                self.parent.active_cells.remove(agent_pos)

        # Start the movement timer
        self.parent.movement_timer = self.parent.root.after(
            self.parent.movement_speed, self.parent.update_moving_cells)


# Function to integrate MARL with the InteractiveGrid application
def integrate_marl_with_app(app):
    """
    Integrate MARL with the InteractiveGrid application.

    Args:
        app: The InteractiveGrid application instance

    Returns:
        MARLIntegration instance
    """
    # Create MARL integration
    marl_integration = MARLIntegration(app)

    # Store reference in the app
    app.marl_integration = marl_integration

    return marl_integration
