import tkinter as tk
from tkinter import ttk
import threading
import time
import os
import random
import math

class GeneticAlgorithmPathPlanner:
    """Simple Genetic Algorithm for path planning."""
    
    def __init__(self, grid_size, population_size=50, generations=20):
        self.grid_size = grid_size
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = 0.2
        self.crossover_rate = 0.7
        self.elite_size = 5
        
    def plan_paths(self, grid_state, agent_positions, target_positions, max_steps=100):
        """Plan paths for all agents using genetic algorithm."""
        print(f"Planning paths with GA for {len(agent_positions)} agents")
        
        # Initialize result
        result = {
            'success': True,
            'paths': [],
            'fitness': []
        }
        
        # Plan path for each agent
        for i, (start, goal) in enumerate(zip(agent_positions, target_positions)):
            print(f"Planning path for agent {i}: {start} -> {goal}")
            path = self._find_path(grid_state, start, goal, max_steps)
            result['paths'].append(path)
            result['fitness'].append(len(path) if path else 0)
            
        return result
    
    def _find_path(self, grid_state, start, goal, max_steps):
        """Find path for a single agent using genetic algorithm."""
        # Initialize population with random paths
        population = self._initialize_population(start, goal, max_steps)
        
        # Evaluate initial population
        fitness_scores = [self._evaluate_fitness(path, grid_state, goal) for path in population]
        
        # Run genetic algorithm for specified generations
        for generation in range(self.generations):
            # Select parents
            parents = self._select_parents(population, fitness_scores)
            
            # Create new population through crossover and mutation
            new_population = []
            
            # Keep elite individuals
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:self.elite_size]
            for i in elite_indices:
                new_population.append(population[i])
            
            # Fill rest of population with crossover and mutation
            while len(new_population) < self.population_size:
                # Select two parents
                parent1, parent2 = random.sample(parents, 2)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child = self._mutate(child, start, goal)
                
                # Add to new population
                new_population.append(child)
            
            # Update population
            population = new_population
            
            # Evaluate new population
            fitness_scores = [self._evaluate_fitness(path, grid_state, goal) for path in population]
            
            # Print progress
            best_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            print(f"Generation {generation+1}/{self.generations}: Best fitness = {best_fitness}, Avg fitness = {avg_fitness:.2f}")
        
        # Return best path
        best_index = fitness_scores.index(max(fitness_scores))
        best_path = population[best_index]
        
        # Check if path reaches goal
        if best_path and best_path[-1] == goal:
            return best_path
        else:
            # Try to repair path if it doesn't reach goal
            repaired_path = self._repair_path(best_path, grid_state, goal)
            return repaired_path if repaired_path and repaired_path[-1] == goal else None
    
    def _initialize_population(self, start, goal, max_steps):
        """Initialize population with random paths."""
        population = []
        
        for _ in range(self.population_size):
            # Create a random path
            path = [start]
            current = start
            
            for _ in range(random.randint(min(5, max_steps), max_steps)):
                # Get possible moves
                moves = [(current[0]+1, current[1]), (current[0]-1, current[1]), 
                         (current[0], current[1]+1), (current[0], current[1]-1)]
                
                # Filter valid moves
                valid_moves = [move for move in moves if 0 <= move[0] < self.grid_size and 0 <= move[1] < self.grid_size]
                
                if not valid_moves:
                    break
                
                # Choose next move with bias towards goal
                if random.random() < 0.7:  # 70% chance to move towards goal
                    valid_moves.sort(key=lambda move: math.dist(move, goal))
                    next_move = valid_moves[0]
                else:
                    next_move = random.choice(valid_moves)
                
                path.append(next_move)
                current = next_move
                
                # Stop if goal reached
                if current == goal:
                    break
            
            population.append(path)
        
        return population
    
    def _evaluate_fitness(self, path, grid_state, goal):
        """Evaluate fitness of a path."""
        if not path:
            return 0
        
        # Check if path reaches goal
        goal_reached = path[-1] == goal
        
        # Calculate path length penalty
        length_penalty = len(path) / 100.0
        
        # Calculate distance to goal
        distance_to_goal = math.dist(path[-1], goal) if path else float('inf')
        
        # Check for obstacles and collisions
        obstacles = 0
        for pos in path:
            if pos in grid_state and (grid_state[pos]["obstacle"] or grid_state[pos]["active"]):
                obstacles += 1
        
        # Calculate fitness (higher is better)
        fitness = 1000.0 if goal_reached else 0
        fitness -= length_penalty * 10  # Shorter paths are better
        fitness -= distance_to_goal * 50  # Closer to goal is better
        fitness -= obstacles * 100  # Avoid obstacles
        
        return max(0, fitness)
    
    def _select_parents(self, population, fitness_scores):
        """Select parents using tournament selection."""
        parents = []
        
        # Tournament selection
        for _ in range(self.population_size):
            # Select random individuals for tournament
            tournament_size = 5
            tournament_indices = random.sample(range(len(population)), tournament_size)
            
            # Find best individual in tournament
            best_index = max(tournament_indices, key=lambda i: fitness_scores[i])
            
            # Add to parents
            parents.append(population[best_index])
        
        return parents
    
    def _crossover(self, parent1, parent2):
        """Perform crossover between two parents."""
        if len(parent1) < 2 or len(parent2) < 2:
            return parent1.copy() if len(parent1) > len(parent2) else parent2.copy()
        
        # Choose crossover point
        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        
        # Create child
        child = parent1[:crossover_point] + parent2[crossover_point:]
        
        return child
    
    def _mutate(self, path, start, goal):
        """Mutate a path."""
        if len(path) < 2:
            return path
        
        # Choose mutation point
        mutation_point = random.randint(1, len(path) - 1)
        
        # Create new path
        new_path = path[:mutation_point]
        
        # Add random steps
        current = new_path[-1]
        
        for _ in range(random.randint(1, 5)):
            # Get possible moves
            moves = [(current[0]+1, current[1]), (current[0]-1, current[1]), 
                     (current[0], current[1]+1), (current[0], current[1]-1)]
            
            # Filter valid moves
            valid_moves = [move for move in moves if 0 <= move[0] < self.grid_size and 0 <= move[1] < self.grid_size]
            
            if not valid_moves:
                break
            
            # Choose next move with bias towards goal
            if random.random() < 0.7:  # 70% chance to move towards goal
                valid_moves.sort(key=lambda move: math.dist(move, goal))
                next_move = valid_moves[0]
            else:
                next_move = random.choice(valid_moves)
            
            new_path.append(next_move)
            current = next_move
            
            # Stop if goal reached
            if current == goal:
                break
        
        return new_path
    
    def _repair_path(self, path, grid_state, goal):
        """Try to repair a path to reach the goal."""
        if not path:
            return None
        
        # Start from the end of the current path
        new_path = path.copy()
        current = new_path[-1]
        
        # Try to reach goal with A* like approach
        max_repair_steps = 20
        visited = set(path)
        
        for _ in range(max_repair_steps):
            if current == goal:
                return new_path
            
            # Get possible moves
            moves = [(current[0]+1, current[1]), (current[0]-1, current[1]), 
                     (current[0], current[1]+1), (current[0], current[1]-1)]
            
            # Filter valid moves
            valid_moves = []
            for move in moves:
                if (0 <= move[0] < self.grid_size and 0 <= move[1] < self.grid_size and 
                    move not in visited and 
                    (move not in grid_state or 
                     (not grid_state[move]["obstacle"] and not grid_state[move]["active"]))):
                    valid_moves.append(move)
            
            if not valid_moves:
                break
            
            # Choose move closest to goal
            next_move = min(valid_moves, key=lambda move: math.dist(move, goal))
            
            new_path.append(next_move)
            visited.add(next_move)
            current = next_move
        
        return new_path

def integrate_ga_with_app(app):
    """
    Simple integration of Genetic Algorithm with the InteractiveGrid application.
    
    Args:
        app: The InteractiveGrid application instance
    """
    # Create a frame at the bottom of the window
    ga_frame = tk.Frame(app.root, bg="#34495E", padx=10, pady=5)
    ga_frame.pack(fill=tk.X, side=tk.BOTTOM)
    
    # Add a label
    tk.Label(
        ga_frame, 
        text="Genetic Algorithm Path Planning", 
        font=("Arial", 10, "bold"),
        bg="#34495E", 
        fg="white"
    ).pack(side=tk.LEFT, padx=5)
    
    # Add population size input
    tk.Label(
        ga_frame, 
        text="Population:", 
        bg="#34495E", 
        fg="white"
    ).pack(side=tk.LEFT, padx=5)
    
    population_var = tk.IntVar(value=50)
    population_entry = tk.Entry(ga_frame, textvariable=population_var, width=5)
    population_entry.pack(side=tk.LEFT, padx=5)
    
    # Add generations input
    tk.Label(
        ga_frame, 
        text="Generations:", 
        bg="#34495E", 
        fg="white"
    ).pack(side=tk.LEFT, padx=5)
    
    generations_var = tk.IntVar(value=20)
    generations_entry = tk.Entry(ga_frame, textvariable=generations_var, width=5)
    generations_entry.pack(side=tk.LEFT, padx=5)
    
    # Add train button
    train_btn = tk.Button(
        ga_frame, 
        text="Train GA for Path Planning", 
        command=lambda: start_ga_training(app, population_var.get(), generations_var.get()),
        bg="#E74C3C",
        fg="white",
        activebackground="#C0392B"
    )
    train_btn.pack(side=tk.LEFT, padx=10)
    
    # Add use GA button
    use_ga_btn = tk.Button(
        ga_frame, 
        text="Do Shape with GA", 
        command=lambda: do_shape_with_ga(app),
        bg="#9B59B6",
        fg="white",
        activebackground="#8E44AD",
        state=tk.DISABLED
    )
    use_ga_btn.pack(side=tk.LEFT, padx=10)
    
    # Add progress bar
    progress_bar = ttk.Progressbar(ga_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
    progress_bar.pack(side=tk.LEFT, padx=10)
    
    # Add status label
    status_label = tk.Label(ga_frame, text="Not started", bg="#34495E", fg="white")
    status_label.pack(side=tk.LEFT, padx=5)
    
    # Store references
    app.ga_integration = {
        'ga': None,
        'population_var': population_var,
        'generations_var': generations_var,
        'train_btn': train_btn,
        'use_ga_btn': use_ga_btn,
        'progress_bar': progress_bar,
        'status_label': status_label,
        'training_active': False,
        'training_progress': 0
    }
    
    return app.ga_integration

def start_ga_training(app, population_size, generations):
    """Start Genetic Algorithm training."""
    if app.ga_integration['training_active']:
        app.update_status("Training already in progress!")
        return
    
    # Initialize GA
    app.ga_integration['ga'] = GeneticAlgorithmPathPlanner(
        grid_size=app.grid_size,
        population_size=population_size,
        generations=generations
    )
    
    # Reset progress
    app.ga_integration['training_progress'] = 0
    app.ga_integration['progress_bar']['value'] = 0
    app.ga_integration['status_label'].config(text="Starting...")
    
    # Disable buttons
    app.ga_integration['train_btn'].config(state=tk.DISABLED)
    app.ga_integration['use_ga_btn'].config(state=tk.DISABLED)
    app.do_shape_btn.config(state=tk.DISABLED)
    
    # Start training in a separate thread
    app.ga_integration['training_active'] = True
    training_thread = threading.Thread(
        target=run_ga_training,
        args=(app, generations)
    )
    training_thread.daemon = True
    training_thread.start()
    
    # Start progress update
    update_ga_training_progress(app, generations)
    
    app.update_status("GA training started for path planning...")

def run_ga_training(app, generations):
    """Run GA training in a separate thread."""
    try:
        # Simulate training (GA doesn't need pre-training, but we'll simulate it)
        for i in range(generations):
            # Update progress
            progress = int((i + 1) / generations * 100)
            app.ga_integration['training_progress'] = progress
            
            # Simulate work
            time.sleep(0.1)
        
        # Signal completion
        app.root.after(0, lambda: ga_training_completed(app, True))
        
    except Exception as e:
        print(f"Error during GA training: {str(e)}")
        app.root.after(0, lambda: app.update_status(f"Error during GA training: {str(e)}"))
        app.root.after(0, lambda: ga_training_completed(app, False))

def update_ga_training_progress(app, generations):
    """Update the training progress bar and status."""
    if app.ga_integration['training_active']:
        # Update progress bar
        app.ga_integration['progress_bar']['value'] = app.ga_integration['training_progress']
        
        # Update status label
        if app.ga_integration['training_progress'] < 100:
            app.ga_integration['status_label'].config(
                text=f"{app.ga_integration['training_progress']}% complete"
            )
        else:
            app.ga_integration['status_label'].config(text="Finalizing...")
        
        # Schedule next update
        app.root.after(100, lambda: update_ga_training_progress(app, generations))

def ga_training_completed(app, success):
    """Called when training is completed."""
    app.ga_integration['training_active'] = False
    
    # Enable buttons
    app.ga_integration['train_btn'].config(state=tk.NORMAL)
    app.do_shape_btn.config(state=tk.NORMAL)
    
    if success:
        # Enable GA button
        app.ga_integration['use_ga_btn'].config(state=tk.NORMAL)
        
        # Update progress and status
        app.ga_integration['progress_bar']['value'] = 100
        app.ga_integration['status_label'].config(text="Training complete")
        
        # Update parent status
        app.update_status("GA training completed successfully!")
    else:
        # Update status
        app.ga_integration['progress_bar']['value'] = 0
        app.ga_integration['status_label'].config(text="Training failed")
        
        # Update parent status
        app.update_status("GA training failed. See console for details.")

def do_shape_with_ga(app):
    """Use the trained GA system to do the shape."""
    print("\n=== DEBUG: do_shape_with_ga called ===")
    
    if not app.ga_integration['ga']:
        app.update_status("Please train GA first!")
        print("DEBUG: GA not initialized")
        return
    
    if app.movement_started:
        app.update_status("Movement already in progress!")
        print("DEBUG: Movement already in progress")
        return
    
    # Set movement started flag
    app.movement_started = True
    print(f"DEBUG: Movement started flag set to {app.movement_started}")
    
    # Disable buttons
    app.ga_integration['train_btn'].config(state=tk.DISABLED)
    app.ga_integration['use_ga_btn'].config(state=tk.DISABLED)
    app.do_shape_btn.config(state=tk.DISABLED)
    
    # Convert grid to GA format
    grid_state = {}
    for cell in app.cells:
        grid_state[cell] = {
            "obstacle": app.cells[cell]["obstacle"],
            "active": app.cells[cell]["active"]
        }
    
    # Get agent positions and targets
    agent_positions = app.active_cells.copy()
    target_positions = app.target_shape[:len(agent_positions)]
    
    print(f"DEBUG: Agent positions: {agent_positions}")
    print(f"DEBUG: Target positions: {target_positions}")
    
    if not agent_positions:
        app.update_status("No active cells found!")
        print("DEBUG: No active cells found")
        app.movement_started = False
        
        # Re-enable buttons
        app.ga_integration['train_btn'].config(state=tk.NORMAL)
        app.ga_integration['use_ga_btn'].config(state=tk.NORMAL)
        app.do_shape_btn.config(state=tk.NORMAL)
        return
    
    if not target_positions:
        app.update_status("No target shape defined!")
        print("DEBUG: No target shape defined")
        app.movement_started = False
        
        # Re-enable buttons
        app.ga_integration['train_btn'].config(state=tk.NORMAL)
        app.ga_integration['use_ga_btn'].config(state=tk.NORMAL)
        app.do_shape_btn.config(state=tk.NORMAL)
        return
    
    # Plan paths using GA
    app.update_status("Planning paths with GA...")
    print("DEBUG: Planning paths with GA...")
    
    try:
        result = app.ga_integration['ga'].plan_paths(
            grid_state, 
            agent_positions, 
            target_positions, 
            max_steps=100
        )
        
        print(f"DEBUG: GA plan_paths result: {result is not None}")
        if result:
            print(f"DEBUG: Paths found: {len(result['paths'])}")
            print(f"DEBUG: Success: {result['success']}")
            
        # Use the paths to move agents
        if result and result['paths']:
            app.update_status("Starting movement with GA paths")
            print("DEBUG: Starting movement with GA paths")
            
            # Initialize moving cells
            app.moving_cells = {}
            
            # Start all movements
            paths_added = 0
            for i, (agent_pos, target_pos, path) in enumerate(zip(agent_positions, target_positions, result['paths'])):
                if not path:
                    print(f"DEBUG: Empty path for agent {i}")
                    continue
                    
                # Generate a unique ID for this movement
                move_id = f"ga_{i}"
                
                # Store the path and current index
                app.moving_cells[move_id] = (path, 1, target_pos, agent_pos)
                
                # Remove cell from active cells
                if agent_pos in app.active_cells:
                    app.active_cells.remove(agent_pos)
                
                paths_added += 1
                print(f"DEBUG: Added path for agent {i}: {agent_pos} -> {target_pos}, path length: {len(path)}")
            
            print(f"DEBUG: Total paths added: {paths_added}")
            print(f"DEBUG: Moving cells: {len(app.moving_cells)}")
            
            if paths_added > 0:
                # Start the movement timer
                print(f"DEBUG: Starting movement timer with speed: {app.movement_speed}")
                app.movement_timer = app.root.after(app.movement_speed, app.update_moving_cells)
                print(f"DEBUG: Movement timer ID: {app.movement_timer}")
            else:
                app.update_status("No valid paths found!")
                print("DEBUG: No valid paths found")
                app.movement_started = False
                
                # Re-enable buttons
                app.ga_integration['train_btn'].config(state=tk.NORMAL)
                app.ga_integration['use_ga_btn'].config(state=tk.NORMAL)
                app.do_shape_btn.config(state=tk.NORMAL)
        else:
            app.update_status("GA path planning failed!")
            print("DEBUG: GA path planning failed")
            app.movement_started = False
            
            # Re-enable buttons
            app.ga_integration['train_btn'].config(state=tk.NORMAL)
            app.ga_integration['use_ga_btn'].config(state=tk.NORMAL)
            app.do_shape_btn.config(state=tk.NORMAL)
    except Exception as e:
        app.update_status(f"Error in GA path planning: {str(e)}")
        print(f"DEBUG: Exception in GA path planning: {str(e)}")
        import traceback
        traceback.print_exc()
        app.movement_started = False
        
        # Re-enable buttons
        app.ga_integration['train_btn'].config(state=tk.NORMAL)
        app.ga_integration['use_ga_btn'].config(state=tk.NORMAL)
        app.do_shape_btn.config(state=tk.NORMAL)
