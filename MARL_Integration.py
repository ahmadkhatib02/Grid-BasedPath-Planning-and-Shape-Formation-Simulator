import tkinter as tk
from tkinter import ttk, scrolledtext, Frame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import os
import threading
import random

# Import the MARL module
from PathPlanningMARL import PathPlanningMARL

class MARLVisualizer:
    """
    A visualization tool for the Multi-Agent Reinforcement Learning path planning.
    This class provides:
    1. Grid visualization of agents and targets
    2. Training progress visualization
    3. Performance metrics tracking
    4. Interactive controls for training and testing
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Agent Reinforcement Learning Visualizer")
        self.root.geometry("1200x800")
        self.root.configure(bg="#F8F9FA")
        
        # MARL parameters
        self.grid_size = 10
        self.num_agents = 3
        self.training_episodes = 100
        self.max_steps = 50
        self.print_interval = 5
        self.save_interval = 20
        
        # Initialize MARL system
        self.marl = PathPlanningMARL(
            grid_size=self.grid_size, 
            num_agents=self.num_agents,
            epsilon=1.0,  # Start with full exploration
            epsilon_decay=0.995,  # Slow decay for better exploration
            memory_size=10000
        )
        
        # Grid state (will be populated with obstacles)
        self.grid_state = {}
        
        # Training metrics
        self.episode_rewards = []
        self.success_rates = []
        self.episode_lengths = []
        self.epsilon_values = []
        
        # Create the UI
        self.create_ui()
        
        # Initialize the grid
        self.initialize_grid()
        
        # Flag for training thread
        self.training_active = False
        self.training_thread = None
        
    def create_ui(self):
        """Create the user interface with controls and visualization areas."""
        # Main container
        main_frame = Frame(self.root, bg="#F8F9FA")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Left panel for grid and controls
        left_panel = Frame(main_frame, bg="#F8F9FA")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Grid canvas
        grid_frame = Frame(left_panel, bg="#34495E", bd=2, relief=tk.RIDGE)
        grid_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.canvas = tk.Canvas(grid_frame, width=500, height=500, bg="#ECF0F1", bd=0, highlightthickness=0)
        self.canvas.pack(padx=5, pady=5)
        
        # Controls frame
        controls_frame = Frame(left_panel, bg="#F8F9FA", bd=2, relief=tk.RIDGE)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Parameters frame
        params_frame = Frame(controls_frame, bg="#F8F9FA")
        params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Grid size
        grid_size_frame = Frame(params_frame, bg="#F8F9FA")
        grid_size_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(grid_size_frame, text="Grid Size:", bg="#F8F9FA").pack(side=tk.LEFT)
        self.grid_size_var = tk.IntVar(value=self.grid_size)
        grid_size_entry = tk.Entry(grid_size_frame, textvariable=self.grid_size_var, width=5)
        grid_size_entry.pack(side=tk.LEFT, padx=5)
        
        # Number of agents
        agents_frame = Frame(params_frame, bg="#F8F9FA")
        agents_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(agents_frame, text="Number of Agents:", bg="#F8F9FA").pack(side=tk.LEFT)
        self.num_agents_var = tk.IntVar(value=self.num_agents)
        agents_entry = tk.Entry(agents_frame, textvariable=self.num_agents_var, width=5)
        agents_entry.pack(side=tk.LEFT, padx=5)
        
        # Training episodes
        episodes_frame = Frame(params_frame, bg="#F8F9FA")
        episodes_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(episodes_frame, text="Training Episodes:", bg="#F8F9FA").pack(side=tk.LEFT)
        self.episodes_var = tk.IntVar(value=self.training_episodes)
        episodes_entry = tk.Entry(episodes_frame, textvariable=self.episodes_var, width=5)
        episodes_entry.pack(side=tk.LEFT, padx=5)
        
        # Apply button
        apply_btn = tk.Button(
            params_frame, 
            text="Apply Settings", 
            command=self.apply_settings,
            bg="#3498DB",
            fg="white",
            activebackground="#2980B9"
        )
        apply_btn.pack(pady=10)
        
        # Action buttons
        actions_frame = Frame(controls_frame, bg="#F8F9FA")
        actions_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Button style
        button_style = {
            "bg": "#3498DB",
            "fg": "white",
            "activebackground": "#2980B9",
            "relief": tk.RAISED,
            "padx": 10,
            "pady": 5,
            "width": 15
        }
        
        # Add obstacles button
        self.obstacles_btn = tk.Button(
            actions_frame, 
            text="Random Obstacles", 
            command=self.add_random_obstacles,
            **button_style
        )
        self.obstacles_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear obstacles button
        self.clear_btn = tk.Button(
            actions_frame, 
            text="Clear Obstacles", 
            command=self.clear_obstacles,
            **button_style
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Training buttons frame
        training_frame = Frame(controls_frame, bg="#F8F9FA")
        training_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Start training button
        self.train_btn = tk.Button(
            training_frame, 
            text="Start Training", 
            command=self.start_training,
            bg="#2ECC71",
            fg="white",
            activebackground="#27AE60",
            relief=tk.RAISED,
            padx=10,
            pady=5,
            width=15
        )
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        # Stop training button
        self.stop_btn = tk.Button(
            training_frame, 
            text="Stop Training", 
            command=self.stop_training,
            bg="#E74C3C",
            fg="white",
            activebackground="#C0392B",
            relief=tk.RAISED,
            padx=10,
            pady=5,
            width=15,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Test button
        self.test_btn = tk.Button(
            training_frame, 
            text="Test Agents", 
            command=self.test_agents,
            bg="#9B59B6",
            fg="white",
            activebackground="#8E44AD",
            relief=tk.RAISED,
            padx=10,
            pady=5,
            width=15
        )
        self.test_btn.pack(side=tk.LEFT, padx=5)
        
        # Right panel for metrics and logs
        right_panel = Frame(main_frame, bg="#F8F9FA")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Metrics visualization
        metrics_frame = Frame(right_panel, bg="#34495E", bd=2, relief=tk.RIDGE)
        metrics_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create matplotlib figure for metrics
        self.fig, self.axs = plt.subplots(2, 2, figsize=(6, 6))
        self.fig.tight_layout(pad=3.0)
        
        # Configure subplots
        self.reward_plot = self.axs[0, 0]
        self.reward_plot.set_title('Average Reward')
        self.reward_plot.set_xlabel('Episode')
        self.reward_plot.set_ylabel('Reward')
        
        self.success_plot = self.axs[0, 1]
        self.success_plot.set_title('Success Rate')
        self.success_plot.set_xlabel('Episode')
        self.success_plot.set_ylabel('Success Rate')
        
        self.steps_plot = self.axs[1, 0]
        self.steps_plot.set_title('Episode Length')
        self.steps_plot.set_xlabel('Episode')
        self.steps_plot.set_ylabel('Steps')
        
        self.epsilon_plot = self.axs[1, 1]
        self.epsilon_plot.set_title('Exploration Rate (Epsilon)')
        self.epsilon_plot.set_xlabel('Episode')
        self.epsilon_plot.set_ylabel('Epsilon')
        
        # Add the plot to the tkinter window
        self.canvas_metrics = FigureCanvasTkAgg(self.fig, metrics_frame)
        self.canvas_metrics.draw()
        self.canvas_metrics.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Log area
        log_frame = Frame(right_panel, bg="#34495E", bd=2, relief=tk.RIDGE)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Log header
        log_header = Frame(log_frame, bg="#34495E", padx=5, pady=2)
        log_header.pack(fill=tk.X)
        
        tk.Label(
            log_header, 
            text="Training Log", 
            font=("Arial", 10, "bold"),
            bg="#34495E", 
            fg="white"
        ).pack(anchor=tk.W)
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(
            log_frame, 
            width=40, 
            height=15, 
            wrap=tk.WORD,
            bg="#ECF0F1", 
            font=("Consolas", 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def initialize_grid(self):
        """Initialize the grid with default settings."""
        # Calculate cell size
        self.cell_size = min(500 // self.grid_size, 40)
        
        # Clear the canvas
        self.canvas.delete("all")
        
        # Reset grid state
        self.grid_state = {}
        
        # Draw the grid
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x1, y1 = col * self.cell_size, row * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                
                # Initially all cells are empty
                fill_color = "#F5F5F5"  # Light gray
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="#95A5A6")
                
                self.grid_state[(row, col)] = {
                    "rect": rect,
                    "obstacle": False,
                    "active": False
                }
        
        # Log initialization
        self.log("Grid initialized")
        self.log(f"Grid size: {self.grid_size}x{self.grid_size}")
        self.log(f"Number of agents: {self.num_agents}")
        
    def apply_settings(self):
        """Apply new settings and reinitialize."""
        try:
            # Get values from UI
            new_grid_size = self.grid_size_var.get()
            new_num_agents = self.num_agents_var.get()
            new_episodes = self.episodes_var.get()
            
            # Validate
            if new_grid_size < 5 or new_grid_size > 20:
                self.log("Grid size must be between 5 and 20")
                return
                
            if new_num_agents < 1 or new_num_agents > 10:
                self.log("Number of agents must be between 1 and 10")
                return
                
            if new_episodes < 10 or new_episodes > 10000:
                self.log("Episodes must be between 10 and 10000")
                return
            
            # Update values
            self.grid_size = new_grid_size
            self.num_agents = new_num_agents
            self.training_episodes = new_episodes
            
            # Reinitialize MARL system
            self.marl = PathPlanningMARL(
                grid_size=self.grid_size, 
                num_agents=self.num_agents,
                epsilon=1.0,
                epsilon_decay=0.995,
                memory_size=10000
            )
            
            # Reset metrics
            self.episode_rewards = []
            self.success_rates = []
            self.episode_lengths = []
            self.epsilon_values = []
            
            # Update plots
            self.update_plots()
            
            # Reinitialize grid
            self.initialize_grid()
            
            self.log("Settings applied successfully")
            
        except Exception as e:
            self.log(f"Error applying settings: {str(e)}")
    
    def add_random_obstacles(self):
        """Add random obstacles to the grid."""
        # Clear existing obstacles
        self.clear_obstacles()
        
        # Number of obstacles (10-20% of grid)
        num_obstacles = random.randint(
            int(0.1 * self.grid_size * self.grid_size),
            int(0.2 * self.grid_size * self.grid_size)
        )
        
        obstacles_added = 0
        for _ in range(num_obstacles * 2):  # Try twice as many times to place obstacles
            if obstacles_added >= num_obstacles:
                break
                
            row = random.randint(0, self.grid_size - 1)
            col = random.randint(0, self.grid_size - 1)
            cell = (row, col)
            
            # Don't place obstacles at the edges to ensure paths exist
            if row == 0 or row == self.grid_size - 1 or col == 0 or col == self.grid_size - 1:
                continue
                
            if not self.grid_state[cell]["obstacle"]:
                self.grid_state[cell]["obstacle"] = True
                self.canvas.itemconfig(self.grid_state[cell]["rect"], fill="#2C3E50")  # Dark blue-gray
                obstacles_added += 1
        
        self.log(f"Added {obstacles_added} random obstacles")
    
    def clear_obstacles(self):
        """Clear all obstacles from the grid."""
        for cell in self.grid_state:
            if self.grid_state[cell]["obstacle"]:
                self.grid_state[cell]["obstacle"] = False
                self.canvas.itemconfig(self.grid_state[cell]["rect"], fill="#F5F5F5")  # Light gray
        
        self.log("Cleared all obstacles")
    
    def start_training(self):
        """Start the training process in a separate thread."""
        if self.training_active:
            return
            
        self.training_active = True
        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.test_btn.config(state=tk.DISABLED)
        
        # Reset metrics
        self.episode_rewards = []
        self.success_rates = []
        self.episode_lengths = []
        self.epsilon_values = []
        
        # Start training thread
        self.training_thread = threading.Thread(target=self.training_loop)
        self.training_thread.daemon = True
        self.training_thread.start()
        
        self.log("Training started")
    
    def stop_training(self):
        """Stop the training process."""
        if not self.training_active:
            return
            
        self.training_active = False
        self.log("Training stopping... (will complete current episode)")
        
    def training_loop(self):
        """Main training loop that runs in a separate thread."""
        try:
            # Get parameters
            num_episodes = self.training_episodes
            max_steps = self.max_steps
            print_interval = self.print_interval
            
            for episode in range(num_episodes):
                if not self.training_active:
                    break
                    
                # Generate random start and target positions
                agent_positions = []
                target_positions = []
                all_positions = set()
                
                # Generate non-overlapping positions
                for _ in range(self.num_agents):
                    # Find a free position for the agent
                    while True:
                        r = random.randint(0, self.grid_size - 1)
                        c = random.randint(0, self.grid_size - 1)
                        pos = (r, c)
                        
                        if (pos not in all_positions and 
                            not self.grid_state.get(pos, {}).get("obstacle", False)):
                            agent_positions.append(pos)
                            all_positions.add(pos)
                            break
                    
                    # Find a free position for the target
                    while True:
                        r = random.randint(0, self.grid_size - 1)
                        c = random.randint(0, self.grid_size - 1)
                        pos = (r, c)
                        
                        if (pos not in all_positions and 
                            not self.grid_state.get(pos, {}).get("obstacle", False)):
                            target_positions.append(pos)
                            all_positions.add(pos)
                            break
                
                # Plan paths for this episode
                result = self.marl.plan_paths(
                    self.grid_state, 
                    agent_positions, 
                    target_positions, 
                    max_steps=max_steps, 
                    training=True
                )
                
                if result:
                    # Calculate metrics
                    success_rate = sum(result['success']) / self.num_agents
                    avg_reward = sum(result['rewards']) / self.num_agents
                    steps = result['steps']
                    
                    # Store metrics
                    self.episode_rewards.append(avg_reward)
                    self.success_rates.append(success_rate)
                    self.episode_lengths.append(steps)
                    self.epsilon_values.append(self.marl.epsilon)
                    
                    # Update UI periodically
                    if (episode + 1) % print_interval == 0:
                        # Log progress
                        log_msg = (f"Episode {episode + 1}/{num_episodes} - "
                                  f"Success rate: {success_rate:.2f}, "
                                  f"Avg reward: {avg_reward:.2f}, "
                                  f"Steps: {steps}, "
                                  f"Epsilon: {self.marl.epsilon:.4f}")
                        
                        # Update log in main thread
                        self.root.after(0, lambda: self.log(log_msg))
                        
                        # Update plots in main thread
                        self.root.after(0, self.update_plots)
                    
                    # Save models periodically
                    if (episode + 1) % self.save_interval == 0:
                        self.marl.save_models()
                        self.root.after(0, lambda: self.log(f"Models saved at episode {episode + 1}"))
            
            # Final save
            if self.training_active:
                self.marl.save_models()
                self.root.after(0, lambda: self.log("Training completed. Models saved."))
            
            # Update UI when done
            self.root.after(0, self.training_completed)
            
        except Exception as e:
            self.root.after(0, lambda: self.log(f"Training error: {str(e)}"))
            self.root.after(0, self.training_completed)
    
    def training_completed(self):
        """Called when training is completed or stopped."""
        self.training_active = False
        self.train_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.test_btn.config(state=tk.NORMAL)
        self.update_plots()
    
    def test_agents(self):
        """Test the trained agents on a new scenario."""
        # Generate random start and target positions
        agent_positions = []
        target_positions = []
        all_positions = set()
        
        # Generate non-overlapping positions
        for _ in range(self.num_agents):
            # Find a free position for the agent
            while True:
                r = random.randint(0, self.grid_size - 1)
                c = random.randint(0, self.grid_size - 1)
                pos = (r, c)
                
                if (pos not in all_positions and 
                    not self.grid_state.get(pos, {}).get("obstacle", False)):
                    agent_positions.append(pos)
                    all_positions.add(pos)
                    break
            
            # Find a free position for the target
            while True:
                r = random.randint(0, self.grid_size - 1)
                c = random.randint(0, self.grid_size - 1)
                pos = (r, c)
                
                if (pos not in all_positions and 
                    not self.grid_state.get(pos, {}).get("obstacle", False)):
                    target_positions.append(pos)
                    all_positions.add(pos)
                    break
        
        # Clear previous visualization
        self.initialize_grid()
        
        # Mark obstacles
        for cell in self.grid_state:
            if self.grid_state[cell]["obstacle"]:
                self.canvas.itemconfig(self.grid_state[cell]["rect"], fill="#2C3E50")  # Dark blue-gray
        
        # Mark agent positions
        for i, pos in enumerate(agent_positions):
            self.canvas.itemconfig(self.grid_state[pos]["rect"], fill="#FF9500")  # Orange
            self.canvas.create_text(
                pos[1] * self.cell_size + self.cell_size // 2,
                pos[0] * self.cell_size + self.cell_size // 2,
                text=f"A{i+1}",
                fill="black",
                font=("Arial", 9, "bold")
            )
        
        # Mark target positions
        for i, pos in enumerate(target_positions):
            self.canvas.itemconfig(self.grid_state[pos]["rect"], fill="#3498DB")  # Blue
            self.canvas.create_text(
                pos[1] * self.cell_size + self.cell_size // 2,
                pos[0] * self.cell_size + self.cell_size // 2,
                text=f"T{i+1}",
                fill="white",
                font=("Arial", 9, "bold")
            )
        
        # Plan paths
        self.log("Testing agents...")
        result = self.marl.plan_paths(
            self.grid_state, 
            agent_positions, 
            target_positions, 
            max_steps=self.max_steps, 
            training=False
        )
        
        if result:
            # Visualize paths
            path_colors = ["#E74C3C", "#2ECC71", "#9B59B6", "#F39C12", "#1ABC9C", 
                          "#3498DB", "#34495E", "#16A085", "#27AE60", "#2980B9"]
            
            for i, path in enumerate(result['paths']):
                color = path_colors[i % len(path_colors)]
                
                # Draw path
                for j in range(len(path) - 1):
                    start_pos = path[j]
                    end_pos = path[j + 1]
                    
                    # Draw a line between cells
                    self.canvas.create_line(
                        start_pos[1] * self.cell_size + self.cell_size // 2,
                        start_pos[0] * self.cell_size + self.cell_size // 2,
                        end_pos[1] * self.cell_size + self.cell_size // 2,
                        end_pos[0] * self.cell_size + self.cell_size // 2,
                        fill=color,
                        width=2,
                        arrow=tk.LAST
                    )
            
            # Log results
            success_rate = sum(result['success']) / self.num_agents
            avg_reward = sum(result['rewards']) / self.num_agents
            steps = result['steps']
            
            self.log(f"Test results:")
            self.log(f"Success rate: {success_rate:.2f}")
            self.log(f"Average reward: {avg_reward:.2f}")
            self.log(f"Steps taken: {steps}")
            
            for i in range(self.num_agents):
                self.log(f"Agent {i+1}: {'Success' if result['success'][i] else 'Failed'}, Reward: {result['rewards'][i]:.2f}")
        else:
            self.log("Testing failed. Make sure ML libraries are available.")
    
    def update_plots(self):
        """Update the metrics plots."""
        # Clear plots
        for ax in self.axs.flat:
            ax.clear()
        
        # Set titles
        self.axs[0, 0].set_title('Average Reward')
        self.axs[0, 0].set_xlabel('Episode')
        self.axs[0, 0].set_ylabel('Reward')
        
        self.axs[0, 1].set_title('Success Rate')
        self.axs[0, 1].set_xlabel('Episode')
        self.axs[0, 1].set_ylabel('Success Rate')
        
        self.axs[1, 0].set_title('Episode Length')
        self.axs[1, 0].set_xlabel('Episode')
        self.axs[1, 0].set_ylabel('Steps')
        
        self.axs[1, 1].set_title('Exploration Rate (Epsilon)')
        self.axs[1, 1].set_xlabel('Episode')
        self.axs[1, 1].set_ylabel('Epsilon')
        
        # Plot data if available
        if self.episode_rewards:
            episodes = list(range(1, len(self.episode_rewards) + 1))
            
            # Reward plot
            self.axs[0, 0].plot(episodes, self.episode_rewards, 'b-')
            
            # Success rate plot
            self.axs[0, 1].plot(episodes, self.success_rates, 'g-')
            self.axs[0, 1].set_ylim([0, 1.1])
            
            # Episode length plot
            self.axs[1, 0].plot(episodes, self.episode_lengths, 'r-')
            
            # Epsilon plot
            self.axs[1, 1].plot(episodes, self.epsilon_values, 'c-')
            self.axs[1, 1].set_ylim([0, 1.1])
        
        # Adjust layout and redraw
        self.fig.tight_layout(pad=3.0)
        self.canvas_metrics.draw()
    
    def log(self, message):
        """Add a message to the log with timestamp."""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = MARLVisualizer(root)
    root.mainloop()
