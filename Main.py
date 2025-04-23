import math
import tkinter as tk
from tkinter import ttk, Frame, Scale, IntVar, StringVar, OptionMenu, scrolledtext
import heapq
from collections import deque
import random
import time

# Grid settings
GRID_SIZE = 10
CELL_SIZE = 40
GRID_COLOR = "black"
ACTIVE_COLOR = "orange"
INACTIVE_COLOR = "white"
OBSTACLE_COLOR = "gray"
OUTLINE_COLOR = "green"


class InteractiveGrid:
    def __init__(self, root):
        self.root = root
        self.root.title("Form a Shape with Performance Metrics")

        # Initialize speed_var first, before anything tries to use it
        self.speed_var = tk.IntVar(value=100)



        # Main container
        main_frame = Frame(root)
        main_frame.pack(padx=10, pady=10)

        # Left frame for the grid
        left_frame = Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=10)

        # Right frame for controls
        right_frame = Frame(main_frame)
        right_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # Canvas for the grid
        self.canvas = tk.Canvas(left_frame, width=GRID_SIZE * CELL_SIZE, height=GRID_SIZE * CELL_SIZE)
        self.canvas.pack()

        # Initialize data structures
        self.cells = {}
        self.active_cells = []
        self.reserved_cells = set()
        self.visited_cells = set()

        #Parallel Movement
        self.moving_cells = {}
        self.movement_timer = None
        self.parallel_mode = False
        self.parallel_var = tk.BooleanVar(value=False)
        self.deadlock_counter = {}
        self.path_attempts = {}
        self.max_attempts = 3
        self.step_counter = 0

        # Performance metrics
        self.metrics = {
            "A*": {"explored": 0, "path_length": 0, "time": 0, "moves": 0, "success": 0, "failures": 0},
            "BFS": {"explored": 0, "path_length": 0, "time": 0, "moves": 0, "success": 0, "failures": 0},
            "DFS": {"explored": 0, "path_length": 0, "time": 0, "moves": 0, "success": 0, "failures": 0}
        }
        self.current_algorithm = "A*"
        self.total_cells_to_fill = 0
        self.cells_filled = 0
        self.start_time = 0

        # Create controls
        self.create_controls(right_frame)

        # Initialize the grid
        self.initialize_grid()

        # Bind click events
        self.canvas.bind("<Button-1>", self.toggle_obstacle)

        # Status
        self.movement_started = False

        # Initialize the grid FIRST to create target_shape
        self.initialize_grid()
        self.layers = []
        self.current_layer = 0
        self.generate_layers()
        self.current_target = self.layers[self.current_layer] if self.layers else []



    def create_controls(self, parent):
        """Create UI controls."""
        # Shape selection
        shape_frame = Frame(parent)
        shape_frame.pack(pady=5, fill=tk.X)

        tk.Label(shape_frame, text="Shape:").pack(side=tk.LEFT)

        self.rectangle_btn = tk.Button(shape_frame, text="Rectangle", command=self.set_rectangle_shape)
        self.rectangle_btn.pack(side=tk.LEFT, padx=2)

        self.triangle_btn = tk.Button(shape_frame, text="Triangle", command=self.set_triangle_shape)
        self.triangle_btn.pack(side=tk.LEFT, padx=2)

        self.circle_btn = tk.Button(shape_frame, text="Circle", command=self.set_circle_shape)
        self.circle_btn.pack(side=tk.LEFT, padx=2)

        # Algorithm selection
        algo_frame = Frame(parent)
        algo_frame.pack(pady=5, fill=tk.X)

        tk.Label(algo_frame, text="Algorithm:").pack(side=tk.LEFT)

        self.algorithm_var = StringVar(value="A*")
        algorithms = ["A*", "BFS", "DFS"]
        algo_menu = OptionMenu(algo_frame, self.algorithm_var, *algorithms)
        algo_menu.pack(side=tk.LEFT, padx=5)
        self.algorithm_var.trace("w", self.on_algorithm_change)

        # Action buttons
        btn_frame = Frame(parent)
        btn_frame.pack(pady=10, fill=tk.X)

        self.do_shape_btn = tk.Button(btn_frame, text="Do the Shape", command=self.start_movement)
        self.do_shape_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = tk.Button(btn_frame, text="Reset", command=self.reset_grid)
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        # Random obstacles button
        obstacles_frame = Frame(parent)
        obstacles_frame.pack(pady=5, fill=tk.X)

        self.random_obstacles_button = tk.Button(obstacles_frame, text="Random Obstacles",
                                                 command=self.add_random_obstacles)
        self.random_obstacles_button.pack(side=tk.LEFT, padx=5)

        self.clear_obstacles_button = tk.Button(obstacles_frame, text="Clear Obstacles",
                                                command=self.clear_obstacles)
        self.clear_obstacles_button.pack(side=tk.LEFT, padx=5)

        # Status text
        status_frame = Frame(parent)
        status_frame.pack(pady=5, fill=tk.BOTH, expand=True)

        tk.Label(status_frame, text="Status:").pack(anchor=tk.W)

        self.status_text = scrolledtext.ScrolledText(status_frame, width=30, height=10, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        self.status_text.config(state=tk.DISABLED)

        # Metrics display
        metrics_frame = Frame(parent)
        metrics_frame.pack(pady=10, fill=tk.X)

        tk.Label(metrics_frame, text="Performance Metrics:", font=("Arial", 10, "bold")).pack(anchor=tk.W)

        self.metrics_text = scrolledtext.ScrolledText(metrics_frame, width=30, height=15, wrap=tk.WORD)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        self.metrics_text.config(state=tk.DISABLED)

        # Compare button
        compare_frame = Frame(parent)
        compare_frame.pack(pady=5, fill=tk.X)

        self.compare_btn = tk.Button(compare_frame, text="Compare All Algorithms",
                                     command=self.compare_algorithms)
        self.compare_btn.pack(side=tk.LEFT, padx=5)

        # Parallel movement option
        parallel_frame = tk.Frame(parent)
        parallel_frame.pack(pady=5, fill=tk.X)

        parallel_check = tk.Checkbutton(parallel_frame, text="Parallel Movement",
                                        variable=self.parallel_var,
                                        command=self.toggle_parallel_mode)
        parallel_check.pack(side=tk.LEFT, padx=5)



    def toggle_parallel_mode(self):
        """Toggle between sequential and parallel movement modes."""
        self.parallel_mode = self.parallel_var.get()
        mode_name = "Parallel" if self.parallel_mode else "Sequential"
        self.update_status(f"Movement mode changed to {mode_name}")

    def generate_layers(self):
        """Generate layers (outlines) for the target shape."""
        remaining_shape = self.target_shape.copy()
        self.layers = []
        while remaining_shape:
            inner, outline, _ = self.separate_cells(remaining_shape)
            if not outline:
                self.layers.append(remaining_shape)
                break
            self.layers.append(outline)
            remaining_shape = inner


    def on_algorithm_change(self, *args):
        """Handle algorithm change."""
        self.current_algorithm = self.algorithm_var.get()
        self.update_status(f"Algorithm changed to {self.current_algorithm}")

    def update_status(self, message):
        """Update status text."""
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)

    def update_metrics_display(self):
        """Update the metrics display with current data."""
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)

        # For the current algorithm, show detailed metrics
        algo = self.current_algorithm
        self.metrics_text.insert(tk.END, f"Current Algorithm: {algo}\n")
        self.metrics_text.insert(tk.END, f"------------------------\n")

        if self.metrics[algo]["moves"] > 0:
            avg_time = self.metrics[algo]["time"] / self.metrics[algo]["moves"]
            avg_explored = self.metrics[algo]["explored"] / self.metrics[algo]["moves"]
            avg_path = self.metrics[algo]["path_length"] / self.metrics[algo]["moves"]
            success_rate = (self.metrics[algo]["success"] /
                            (self.metrics[algo]["success"] + self.metrics[algo]["failures"])) * 100 if \
                (self.metrics[algo]["success"] + self.metrics[algo]["failures"]) > 0 else 0

            self.metrics_text.insert(tk.END,
                                     f"Cells explored: {self.metrics[algo]['explored']} (Avg: {avg_explored:.2f})\n")
            self.metrics_text.insert(tk.END,
                                     f"Path length: {self.metrics[algo]['path_length']} (Avg: {avg_path:.2f})\n")
            self.metrics_text.insert(tk.END, f"Time: {self.metrics[algo]['time']:.4f}s (Avg: {avg_time:.4f}s)\n")
            self.metrics_text.insert(tk.END, f"Success rate: {success_rate:.1f}%\n")

            if self.movement_started:
                # Show progress for current shape
                progress = (self.cells_filled / self.total_cells_to_fill) * 100 if self.total_cells_to_fill > 0 else 0
                elapsed = time.time() - self.start_time
                self.metrics_text.insert(tk.END, f"\nCurrent progress: {progress:.1f}%\n")
                self.metrics_text.insert(tk.END, f"Time elapsed: {elapsed:.2f}s\n")
        else:
            self.metrics_text.insert(tk.END, "No data yet for this algorithm.\n")

        self.metrics_text.insert(tk.END, f"\nSummary of All Algorithms:\n")
        self.metrics_text.insert(tk.END, f"------------------------\n")

        # Show summary for all algorithms
        for alg in ["A*", "BFS", "DFS"]:
            if self.metrics[alg]["moves"] > 0:
                avg_time = self.metrics[alg]["time"] / self.metrics[alg]["moves"]
                success_rate = (self.metrics[alg]["success"] /
                                (self.metrics[alg]["success"] + self.metrics[alg]["failures"])) * 100 if \
                    (self.metrics[alg]["success"] + self.metrics[alg]["failures"]) > 0 else 0
                self.metrics_text.insert(tk.END, f"{alg}: Avg time: {avg_time:.4f}s, Success: {success_rate:.1f}%\n")
            else:
                self.metrics_text.insert(tk.END, f"{alg}: No data yet\n")

        self.metrics_text.config(state=tk.DISABLED)

    def initialize_grid(self):
        """Initialize the grid with default settings."""
        # Clear the canvas
        self.canvas.delete("all")

        # Reset data structures
        self.cells = {}
        self.active_cells = []
        self.reserved_cells = set()
        self.visited_cells = set()
        self.cells_filled = 0

        # Draw the grid
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x1, y1 = col * CELL_SIZE, row * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE

                # Bottom two rows start as active cells
                is_active = (row >= GRID_SIZE - 2)

                fill_color = ACTIVE_COLOR if is_active else INACTIVE_COLOR
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline=GRID_COLOR)

                self.cells[(row, col)] = {
                    "rect": rect,
                    "active": is_active,
                    "obstacle": False
                }

                if is_active:
                    self.active_cells.append((row, col))

        # Set default shape (rectangle)
        self.target_shape = self.define_target_rectangle()
        self.inner_cells, self.outline_cells, self.corner_cells = self.separate_cells()
        self.draw_green_outline()

        # Update total cells to fill
        self.total_cells_to_fill = len(self.target_shape)

        # Update status
        self.update_status("Grid initialized with rectangle shape.")
        self.movement_started = False

        # Update metrics display
        self.update_metrics_display()

    def define_target_rectangle(self):
        """Creates a 5x4 rectangle in the grid center (rows 3–6, columns 3–7)."""
        mid_row, mid_col = GRID_SIZE // 2, GRID_SIZE // 2
        return [(mid_row - 2 + r, mid_col - 2 + c) for r in range(4) for c in range(5)]

    def define_target_triangle(self):
        """
        Creates a large triangle placed two rows above the bottom two orange cells.
        The triangle has:
        - Base row: row 6 with 8 cells (columns 1 to 8)
        - Next row: row 5 with 6 cells (columns 2 to 7)
        - Next row: row 4 with 4 cells (columns 3 to 6)
        - Top row: row 3 with 2 cells (columns 4 to 5)
        """
        shape = []
        # Base row: row 6, columns 1..8
        for c in range(1, 9):
            shape.append((6, c))
        # Next row: row 5, columns 2..7
        for c in range(2, 8):
            shape.append((5, c))
        # Next row: row 4, columns 3..6
        for c in range(3, 7):
            shape.append((4, c))
        # Top row: row 3, columns 4..5
        for c in range(4, 6):
            shape.append((3, c))
        return shape

    def define_target_circle(self):
        """
        Creates a circle (oval) shape with a custom pattern spanning 6 rows.
        The pattern (relative to the shape's own rows) is:
        - Row 1: 2 cells at columns 4 and 5
        - Row 2: 4 cells at columns 3,4,5,6
        - Row 3: 2 cells (columns 2,3), then a gap (columns 4,5), then 2 cells (columns 6,7)
        - Row 4: same as Row 3
        - Row 5: 4 cells at columns 3,4,5,6
        - Row 6: 2 cells at columns 4 and 5
        These rows are mapped to grid rows 1 to 6.
        """
        shape = []
        # Row 1 (grid row 1): 2 cells at columns 4 and 5
        for c in [4, 5]:
            shape.append((1, c))
        # Row 2 (grid row 2): 4 cells at columns 3,4,5,6
        for c in range(3, 7):
            shape.append((2, c))
        # Row 3 (grid row 3): cells at columns 2,3 and 6,7
        for c in [2, 3, 6, 7]:
            shape.append((3, c))
        # Row 4 (grid row 4): same as row 3
        for c in [2, 3, 6, 7]:
            shape.append((4, c))
        # Row 5 (grid row 5): 4 cells at columns 3,4,5,6
        for c in range(3, 7):
            shape.append((5, c))
        # Row 6 (grid row 6): 2 cells at columns 4 and 5
        for c in [4, 5]:
            shape.append((6, c))
        return shape

    def set_rectangle_shape(self):
        """Switch to the rectangle shape (if movement hasn't started)."""
        if self.movement_started:
            return
        self.remove_green_outline()
        self.target_shape = self.define_target_rectangle()
        self.inner_cells, self.outline_cells, self.corner_cells = self.separate_cells()
        self.draw_green_outline()
        self.total_cells_to_fill = len(self.target_shape)
        self.update_status("Shape changed to rectangle.")

    def set_triangle_shape(self):
        """Switch to the triangle shape (if movement hasn't started)."""
        if self.movement_started:
            return
        self.remove_green_outline()
        self.target_shape = self.define_target_triangle()
        self.inner_cells, self.outline_cells, self.corner_cells = self.separate_cells()
        self.draw_green_outline()
        self.total_cells_to_fill = len(self.target_shape)
        self.update_status("Shape changed to triangle.")

    def set_circle_shape(self):
        """Switch to the circle shape (if movement hasn't started)."""
        if self.movement_started:
            return
        self.remove_green_outline()
        self.target_shape = self.define_target_circle()
        self.inner_cells, self.outline_cells, self.corner_cells = self.separate_cells()
        self.draw_green_outline()
        self.total_cells_to_fill = len(self.target_shape)
        self.update_status("Shape changed to circle.")

    def separate_cells(self, target_cells=None):
        if target_cells is None:
            target_cells = self.target_shape  # Fallback for legacy calls
        if not target_cells:
            return [], [], []

        min_row = min(c[0] for c in target_cells)
        max_row = max(c[0] for c in target_cells)
        min_col = min(c[1] for c in target_cells)
        max_col = max(c[1] for c in target_cells)

        corners = [
            (min_row, min_col), (min_row, max_col),
            (max_row, max_col), (max_row, min_col)
        ]

        outline = []
        inner = []
        for cell in target_cells:
            if cell in corners:
                continue
            if (cell[0] in (min_row, max_row)) or (cell[1] in (min_col, max_col)):
                outline.append(cell)
            else:
                inner.append(cell)

        return inner, outline, corners

    def draw_green_outline(self):
        """Draws the green outline for the current target shape."""
        for cell in self.target_shape:
            row, col = cell
            if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                x1, y1 = col * CELL_SIZE, row * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill=OUTLINE_COLOR, outline=GRID_COLOR)
                self.cells[cell]["outline_rect"] = rect

    def remove_green_outline(self):
        """Removes the green outline for the current target shape."""
        for cell in self.target_shape:
            if cell in self.cells and "outline_rect" in self.cells[cell]:
                self.canvas.delete(self.cells[cell]["outline_rect"])
                if "outline_rect" in self.cells[cell]:
                    del self.cells[cell]["outline_rect"]

    def toggle_obstacle(self, event):
        """Toggles a cell's obstacle status on click (disabled once movement starts)."""
        if self.movement_started:
            return

        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE

        if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
            return

        cell = (row, col)

        # Don't allow toggling active cells or target shape cells
        if cell in self.cells and not self.cells[cell]["active"] and cell not in self.target_shape:
            if self.cells[cell]["obstacle"]:
                self.canvas.itemconfig(self.cells[cell]["rect"], fill=INACTIVE_COLOR)
                self.cells[cell]["obstacle"] = False
                self.update_status(f"Removed obstacle at ({row}, {col})")
            else:
                self.canvas.itemconfig(self.cells[cell]["rect"], fill=OBSTACLE_COLOR)
                self.cells[cell]["obstacle"] = True
                self.update_status(f"Added obstacle at ({row}, {col})")

    def add_random_obstacles(self):
        """Add random obstacles to the grid."""
        if self.movement_started:
            return

        # Clear existing obstacles
        self.clear_obstacles()

        # Number of obstacles to add (adjust as needed)
        num_obstacles = random.randint(10, 30)

        for _ in range(num_obstacles):
            row = random.randint(0, GRID_SIZE - 1)
            col = random.randint(0, GRID_SIZE - 1)
            cell = (row, col)

            # Don't add obstacles in active cells or target shape
            if cell in self.active_cells or cell in self.target_shape:
                continue

            self.cells[cell]["obstacle"] = True
            self.canvas.itemconfig(self.cells[cell]["rect"], fill=OBSTACLE_COLOR)

        self.update_status("Added random obstacles.")

    def clear_obstacles(self):
        """Clear all obstacles from the grid."""
        if self.movement_started:
            return

        for cell in self.cells:
            if self.cells[cell]["obstacle"]:
                self.cells[cell]["obstacle"] = False
                fill_color = ACTIVE_COLOR if cell in self.active_cells else INACTIVE_COLOR
                self.canvas.itemconfig(self.cells[cell]["rect"], fill=fill_color)

        self.update_status("Cleared all obstacles.")

    def reset_grid(self):
        """
        Perform a hard reset of the grid, immediately stopping any movement with no auto-restart.
        """
        # 1. Stop all ongoing movements
        self.movement_started = False

        # 2. Cancel ALL pending timers
        if hasattr(self, "movement_timer") and self.movement_timer:
            self.root.after_cancel(self.movement_timer)
            self.movement_timer = None

        # Cancel any other potential callbacks that might be in the queue
        for after_id in self.root.tk.call('after', 'info'):
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass

        # 3. Reset all movement-related data structures
        self.moving_cells = {}
        self.deadlock_counter = {}
        self.path_attempts = {}
        self.step_counter = 0
        self.reserved_cells = set()
        self.visited_cells = set()

        # 4. Enable the "Do the Shape" button
        if hasattr(self, "do_shape_btn"):
            self.do_shape_btn.config(state=tk.NORMAL)

        # 5. Clear the canvas
        self.canvas.delete("all")

        # 6. Reinitialize the grid
        self.initialize_grid()

        # 7. Rebind click events (in case they were unbound during movement)
        self.canvas.bind("<Button-1>", self.toggle_obstacle)

        # 8. Update status
        self.update_status("Grid has been reset.")

    def start_movement(self):
        """Start the movement of active cells to form the target shape."""
        if self.movement_started or not self.active_cells:
            return

        self.movement_started = True
        self.do_shape_btn.config(state=tk.DISABLED)

        # Clear target shape overlay
        self.remove_green_outline()

        # Initialize metrics for this run
        self.start_time = time.time()
        self.cells_filled = 0
        self.moving_cells = {}

        # Update status based on selected mode
        mode = "parallel" if self.parallel_mode else "sequential"
        self.update_status(f"Starting {mode} movement with {self.current_algorithm} algorithm.")

        if self.parallel_mode:
            # Start parallel movement
            self.movement_timer = self.root.after(50, self.process_parallel_movements)
        else:
            # Original sequential movement
            self.update_status(f"Starting movement with {self.current_algorithm} algorithm.")
            self.move_next_square()

    def process_parallel_movements(self):
        """
        Main movement process for parallel mode with improved cell targeting.
        Ensures all cells in shapes, especially circles, are properly filled.
        """
        # Increment global step counter
        self.step_counter += 1

        # Update all currently moving cells
        self.update_moving_cells()

        # Run deadlock detection and resolution every 5 steps
        if self.step_counter % 5 == 0:
            self.detect_and_resolve_deadlocks()

        # Check if current layer is filled
        current_layer_filled = all(self.cells[cell]["active"] for cell in self.current_target)
        if current_layer_filled and self.current_layer < len(self.layers) - 1:
            # Move to next layer
            self.current_layer += 1
            self.current_target = self.layers[self.current_layer]
            # Reactivate previous layer's cells
            for cell in self.layers[self.current_layer - 1]:
                if self.cells[cell]["active"]:
                    self.active_cells.append(cell)
            # Reset path attempts
            self.path_attempts.clear()

        # Clear path attempts every 30 steps to allow retrying
        if self.step_counter % 30 == 0:
            self.path_attempts.clear()

            # Specifically check for unfilled cells and mark them for targeting
            unfilled_targets = [cell for cell in self.target_shape if not self.cells[cell]["active"]]
            if unfilled_targets:
                self.update_status(f"Prioritizing {len(unfilled_targets)} unfilled cells...")
                # Reset attempt counters to force retrying
                for target in unfilled_targets:
                    if target in self.path_attempts:
                        self.path_attempts[target] = 0

        # Check if we're done
        if not self.active_cells and not self.moving_cells:
            self.finish_movement(success=False)
            return

        # Identify remaining targets
        remaining_targets = [cell for cell in self.target_shape if not self.cells[cell]["active"]]
        if not remaining_targets and not self.moving_cells:
            self.finish_movement(success=True)
            return

        # Start new movements for all available active cells
        if remaining_targets and self.active_cells:
            self.start_new_movements(remaining_targets)

        # Continue the movement process
        self.movement_timer = self.root.after(50, self.process_parallel_movements)

    def start_new_movements(self, remaining_targets):
        """
        Start new movements for active cells with improved prioritization to ensure
        full shape filling, especially problematic cells like row 5, column 4 in circle.
        """
        # Get active cells that aren't already moving
        available_active_cells = [
            cell for cell in self.active_cells
            if cell not in [info[3] for info in self.moving_cells.values() if len(info) > 3]
        ]

        if not available_active_cells or not remaining_targets:
            return

        # Create a special priority list - cells that have been 'stuck' and
        # need special attention (like the problematic cell at (5,4))
        priority_targets = []
        for target in remaining_targets:
            # Check if this is one of our problematic cells
            row, col = target
            # Specifically prioritize cells that are typically problematic in the circle
            if (row == 5 and col == 4) or (row == 4 and col == 5) or (row == 4 and col == 4) or (row == 5 and col == 5):
                priority_targets.append(target)
            # Also prioritize any cell that has surrounding filled cells
            filled_neighbors = 0
            for neighbor in self.get_neighbors(target):
                if neighbor in self.target_shape and self.cells[neighbor]["active"]:
                    filled_neighbors += 1
            # If a cell has 3+ filled neighbors, it might be getting trapped
            if filled_neighbors >= 3:
                priority_targets.append(target)

        # Map to track which cells will be used by new paths
        newly_reserved_cells = set()

        # List of movements to actually start after planning
        movements_to_start = []

        # First handle priority targets
        if priority_targets:
            self.update_status(f"Handling {len(priority_targets)} priority targets")
            # Sort active cells by distance to priority targets
            sorted_active_cells = sorted(
                available_active_cells,
                key=lambda cell: min([math.dist(cell, target) for target in priority_targets])
            )

            # Try to assign active cells to priority targets first
            for priority_target in priority_targets:
                if priority_target not in remaining_targets:
                    continue

                # Find closest active cell that can reach this target
                best_path = None
                best_active = None
                for active_cell in sorted_active_cells[:]:  # Make a copy to allow removal
                    path = self.find_path(active_cell, priority_target)
                    if path:
                        # Check path conflicts
                        conflict = False
                        for step in path:
                            if step in newly_reserved_cells:
                                conflict = True
                                break

                        if not conflict:
                            best_path = path
                            best_active = active_cell
                            break

                if best_path and best_active:
                    # Reserve path
                    for step in best_path:
                        newly_reserved_cells.add(step)

                    # Add movement
                    movements_to_start.append((best_active, priority_target, best_path))

                    # Remove from consideration
                    if best_active in sorted_active_cells:
                        sorted_active_cells.remove(best_active)
                        available_active_cells.remove(best_active)
                    remaining_targets.remove(priority_target)

                    # Update tracking
                    if priority_target not in self.path_attempts:
                        self.path_attempts[priority_target] = 0
                    self.path_attempts[priority_target] += 1

        # Process remaining targets normally
        # Sort active cells by distance to any target
        sorted_active_cells = sorted(
            available_active_cells,
            key=lambda cell: min([math.dist(cell, target) for target in remaining_targets])
        )

        # Process cells in sorted order
        for active_cell in sorted_active_cells:
            # Skip if we've used all targets
            if not remaining_targets:
                break

            # Calculate target costs excluding newly reserved paths
            target_costs = {}
            for target in remaining_targets:
                # Skip targets that have been attempted too many times
                if target in self.path_attempts and self.path_attempts[target] >= self.max_attempts:
                    continue

                path = self.find_path(active_cell, target)
                if path:
                    # Check if path conflicts with newly reserved cells
                    conflict = False
                    for step in path:
                        if step in newly_reserved_cells:
                            conflict = True
                            break

                    if not conflict:
                        target_costs[target] = len(path)

            if not target_costs:
                continue  # No available targets for this cell

            # Select the best target (prioritize shortest paths for remaining cells)
            best_target = min(target_costs, key=lambda k: target_costs[k])
            path = self.find_path(active_cell, best_target)

            if path:
                # Reserve all cells in this path
                for step in path:
                    newly_reserved_cells.add(step)

                # Track this movement
                movements_to_start.append((active_cell, best_target, path))

                # Remove this target from consideration
                remaining_targets.remove(best_target)

                # Increment attempt counter
                if best_target not in self.path_attempts:
                    self.path_attempts[best_target] = 0
                self.path_attempts[best_target] += 1

        # Now actually start all the planned movements
        for active_cell, target, path in movements_to_start:
            # Remove cell from active cells
            self.active_cells.remove(active_cell)

            # Update metrics
            self.metrics[self.current_algorithm]["path_length"] += len(path)
            self.metrics[self.current_algorithm]["moves"] += 1

            # Reserve cells along the path
            self.reserve_cells(path)

            # Follow the path (animate movement)
            self.follow_path(path, active_cell, target)

    def update_moving_cells(self):
        """Update the position of all currently moving cells with synchronized movement."""
        # First pass - collect all planned moves and detect collisions
        completed_moves = []
        next_positions = {}
        collision_cells = set()

        for move_id, (path, index, target, start_cell) in self.moving_cells.items():
            if index >= len(path):
                completed_moves.append(move_id)
                continue

            next_pos = path[index]

            # If two cells would move to the same position, mark as collision
            if next_pos in next_positions:
                collision_cells.add(next_pos)
                # Mark both movements for deadlock increment
                self.deadlock_counter[move_id] = self.deadlock_counter.get(move_id, 0) + 1
                self.deadlock_counter[next_positions[next_pos]] = self.deadlock_counter.get(next_positions[next_pos],
                                                                                            0) + 1
            else:
                next_positions[next_pos] = move_id

        # Second pass - collect all movements that will be executed
        moves_to_execute = []
        for move_id, (path, index, target, start_cell) in self.moving_cells.items():
            if index >= len(path) or move_id in completed_moves:
                continue

            next_pos = path[index]
            # Skip moves to collision cells
            if next_pos in collision_cells:
                self.deadlock_counter[move_id] = self.deadlock_counter.get(move_id, 0) + 1
                continue

            # Get current position
            prev_cell = path[index - 1] if index > 0 else start_cell

            # Add to execution list
            moves_to_execute.append((move_id, prev_cell, next_pos))

            # Move to next step in path
            self.moving_cells[move_id] = (path, index + 1, target, start_cell)

            # Reset deadlock counter since movement happened
            self.deadlock_counter[move_id] = 0

        # Third pass - execute all valid moves at once
        # First deactivate all previous positions
        for move_id, prev_cell, next_pos in moves_to_execute:
            self.deactivate_cell(prev_cell)
            self.unreserve_cell(prev_cell)

        # Then activate all new positions
        for move_id, prev_cell, next_pos in moves_to_execute:
            self.activate_cell(next_pos)

        # Process completed movements
        for move_id in completed_moves:
            if move_id in self.moving_cells:
                path, index, target, start_cell = self.moving_cells[move_id]

                # Activate the target cell
                self.activate_cell(target)
                self.cells[target]["active"] = True
                self.unreserve_cell(target)

                # Clear attempt counter for this target
                if target in self.path_attempts:
                    del self.path_attempts[target]

                # Increment filled cells counter
                self.cells_filled += 1

                # Update metrics display
                self.update_metrics_display()

                # Remove from tracking
                if move_id in self.deadlock_counter:
                    del self.deadlock_counter[move_id]
                del self.moving_cells[move_id]

    def finish_movement(self, success):
        """Handle the end of all movements."""
        elapsed_time = time.time() - self.start_time

        if success:
            self.update_status(f"Target shape formation complete in {elapsed_time:.2f}s!")
            self.metrics[self.current_algorithm]["success"] += 1
        else:
            self.update_status("Movement failed: No more active cells available.")
            self.metrics[self.current_algorithm]["failures"] += 1

        self.metrics[self.current_algorithm]["time"] += elapsed_time
        self.update_metrics_display()

        self.movement_started = False
        self.do_shape_btn.config(state=tk.NORMAL)

        # Cancel any pending timer
        if self.movement_timer:
            self.root.after_cancel(self.movement_timer)
            self.movement_timer = None

    def detect_and_resolve_deadlocks(self):
        """Enhanced deadlock detection and resolution."""
        deadlocked_movements = []

        # Check for cells that haven't moved
        for move_id, (path, index, target, start_cell) in self.moving_cells.items():
            if move_id in self.deadlock_counter:
                self.deadlock_counter[move_id] += 1

                # If stuck for too long (shorter time than before)
                if self.deadlock_counter[move_id] > 10:  # About 0.5 seconds
                    deadlocked_movements.append(move_id)

        # If more than 30% of movements are deadlocked, do a major reset
        if deadlocked_movements and len(deadlocked_movements) > 0.3 * len(self.moving_cells):
            self.update_status("Major deadlock detected, resetting all movements")

            # Reset all movements
            for move_id in list(self.moving_cells.keys()):
                self.abort_movement(move_id)

            # Clear all reservations to start fresh
            self.reserved_cells.clear()
        else:
            # Just abort individual deadlocked movements
            for move_id in deadlocked_movements:
                self.update_status(f"Resetting stuck movement {move_id}")
                self.abort_movement(move_id)

    def abort_movement(self, move_id):
        """Abort a movement and return the cell to available active cells."""
        if move_id not in self.moving_cells:
            return

        path, index, target, start_cell = self.moving_cells[move_id]

        # Get current position
        current_pos = path[index - 1] if index > 0 else start_cell

        # Unreserve path
        for cell in path[index:]:
            self.unreserve_cell(cell)

        # Return cell to active cells
        self.active_cells.append(current_pos)

        # Remove from moving cells
        del self.moving_cells[move_id]
        if move_id in self.deadlock_counter:
            del self.deadlock_counter[move_id]

    def move_next_square(self):
        """Core movement logic with dynamic hardest-target selection."""
        if not self.active_cells:
            self.update_status("No more active cells available.")

            # Update metrics for failure
            self.metrics[self.current_algorithm]["failures"] += 1
            self.update_metrics_display()

            self.movement_started = False
            self.do_shape_btn.config(state=tk.NORMAL)
            return

        remaining_targets = [cell for cell in self.target_shape if not self.cells[cell]["active"]]
        if not remaining_targets:
            elapsed_time = time.time() - self.start_time
            self.update_status(f"Target shape formation complete in {elapsed_time:.2f}s!")

            # Update metrics for success
            self.metrics[self.current_algorithm]["success"] += 1
            self.metrics[self.current_algorithm]["time"] += elapsed_time
            self.update_metrics_display()

            self.movement_started = False
            self.do_shape_btn.config(state=tk.NORMAL)
            return

        # Calculate minimal path lengths for each target
        target_costs = {}
        for target in remaining_targets:
            min_length = float('inf')
            for active in self.active_cells:
                path = self.find_path(active, target)
                if path:
                    min_length = min(min_length, len(path))
            if min_length != float('inf'):
                target_costs[target] = min_length

        if not target_costs:
            self.update_status("No reachable targets left")

            # Update metrics for failure
            self.metrics[self.current_algorithm]["failures"] += 1
            self.update_metrics_display()

            self.movement_started = False
            self.do_shape_btn.config(state=tk.NORMAL)
            return

        # Select the hardest target (with the maximum minimal path length)
        hardest_target = max(target_costs, key=lambda k: target_costs[k])

        # Find the active cell with the shortest path to the hardest target
        shortest_path = None
        shortest_length = float('inf')
        selected_active = None

        for active in self.active_cells:
            path = self.find_path(active, hardest_target)
            if path and len(path) < shortest_length:
                shortest_length = len(path)
                shortest_path = path
                selected_active = active

        if not selected_active:
            self.root.after(50, self.move_next_square)
            return

        self.active_cells.remove(selected_active)

        if shortest_path:
            # Update path length metric
            self.metrics[self.current_algorithm]["path_length"] += len(shortest_path)
            self.metrics[self.current_algorithm]["moves"] += 1

            self.reserve_cells(shortest_path)
            self.follow_path(shortest_path, selected_active, hardest_target)
        else:
            self.active_cells.append(selected_active)
            self.root.after(500, self.move_next_square)

    def find_path(self, start, goal):
        """Find path using the selected algorithm."""
        algorithm = self.current_algorithm

        # Clear previous visited cells visualization
        self.clear_visited_cells()

        # Count of cells explored during this search
        explored_count = 0

        if algorithm == "A*":
            path, explored_count = self.a_star(start, goal)
        elif algorithm == "BFS":
            path, explored_count = self.bfs_search(start, goal)
        elif algorithm == "DFS":
            path, explored_count = self.dfs_search(start, goal)
        else:
            # Default to A*
            path, explored_count = self.a_star(start, goal)

        # Update explored cells metric
        self.metrics[algorithm]["explored"] += explored_count

        return path

    def a_star(self, start, goal):
        """A* pathfinding implementation with obstacle avoidance."""
        start_time = time.time()
        explored_count = 0

        def heuristic(a, b):
            return math.dist(a, b)

        open_set = []
        heapq.heappush(open_set, (0, start))
        g_score = {start: 0}
        came_from = {}

        while open_set:
            _, current = heapq.heappop(open_set)
            explored_count += 1

            # Mark as visited (for visualization)
            if current != start and current != goal:
                self.mark_visited(current)

            if current == goal:
                # Calculate algorithm execution time
                execution_time = time.time() - start_time
                self.metrics[self.current_algorithm]["time"] += execution_time

                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1], explored_count

            for neighbor in self.get_neighbors(current):
                if (self.cells[neighbor]["obstacle"] or
                        self.cells[neighbor]["active"] or
                        neighbor in self.reserved_cells):
                    continue

                move_cost = 1 if abs(neighbor[0] - current[0]) + abs(neighbor[1] - current[1]) == 1 else math.sqrt(2)
                tentative_g_score = g_score.get(current, float('inf')) + move_cost

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

        # Calculate algorithm execution time even if path not found
        execution_time = time.time() - start_time
        self.metrics[self.current_algorithm]["time"] += execution_time

        return None, explored_count

    def bfs_search(self, start, goal):
        """Breadth-First Search implementation."""
        start_time = time.time()
        explored_count = 0

        queue = deque([[start]])
        visited = {start}

        while queue:
            path = queue.popleft()
            current = path[-1]
            explored_count += 1

            # Mark as visited (for visualization)
            if current != start and current != goal:
                self.mark_visited(current)

            if current == goal:
                # Calculate algorithm execution time
                execution_time = time.time() - start_time
                self.metrics[self.current_algorithm]["time"] += execution_time

                return path, explored_count

            for neighbor in self.get_neighbors(current):
                if (neighbor in visited or
                        self.cells[neighbor]["obstacle"] or
                        self.cells[neighbor]["active"] or
                        neighbor in self.reserved_cells):
                    continue

                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

        # Calculate algorithm execution time even if path not found
        execution_time = time.time() - start_time
        self.metrics[self.current_algorithm]["time"] += execution_time

        return None, explored_count

    def dfs_search(self, start, goal):
        """Depth-First Search implementation."""
        start_time = time.time()
        explored_count = 0

        stack = [[start]]
        visited = {start}

        while stack:
            path = stack.pop()
            current = path[-1]
            explored_count += 1

            # Mark as visited (for visualization)
            if current != start and current != goal:
                self.mark_visited(current)

            if current == goal:
                # Calculate algorithm execution time
                execution_time = time.time() - start_time
                self.metrics[self.current_algorithm]["time"] += execution_time

                return path, explored_count

            # Reverse neighbors to maintain a more natural DFS order
            neighbors = list(self.get_neighbors(current))
            neighbors.reverse()

            for neighbor in neighbors:
                if (neighbor in visited or
                        self.cells[neighbor]["obstacle"] or
                        self.cells[neighbor]["active"] or
                        neighbor in self.reserved_cells):
                    continue

                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                stack.append(new_path)

        # Calculate algorithm execution time even if path not found
        execution_time = time.time() - start_time
        self.metrics[self.current_algorithm]["time"] += execution_time

        return None, explored_count

    def get_neighbors(self, cell):
        """Returns valid neighbor cells with diagonal moves allowed only if adjacent cells are free."""
        row, col = cell
        neighbors = []

        # Cardinal moves
        cardinal_moves = [
            (row - 1, col), (row + 1, col),
            (row, col - 1), (row, col + 1)
        ]

        for r, c in cardinal_moves:
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                neighbors.append((r, c))

        # Diagonal moves (only allowed if adjacent cardinal cells are free)
        diagonal_moves = [
            (row - 1, col - 1), (row - 1, col + 1),
            (row + 1, col - 1), (row + 1, col + 1)
        ]

        for i, (r, c) in enumerate(diagonal_moves):
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                # Check if the adjacent cardinal cells are free
                if i == 0:  # NW
                    adj1, adj2 = (row - 1, col), (row, col - 1)
                elif i == 1:  # NE
                    adj1, adj2 = (row - 1, col), (row, col + 1)
                elif i == 2:  # SW
                    adj1, adj2 = (row + 1, col), (row, col - 1)
                else:  # SE
                    adj1, adj2 = (row + 1, col), (row, col + 1)

                # Allow diagonal moves only if both adjacent cells are valid and not obstacles
                if (0 <= adj1[0] < GRID_SIZE and 0 <= adj1[1] < GRID_SIZE and
                        0 <= adj2[0] < GRID_SIZE and 0 <= adj2[1] < GRID_SIZE and
                        not self.cells[adj1]["obstacle"] and
                        not self.cells[adj2]["obstacle"]):
                    neighbors.append((r, c))

        return neighbors

    def mark_visited(self, cell):
        """Mark a cell as visited during search (for visualization)."""
        self.visited_cells.add(cell)


    def clear_visited_cells(self):
        """Clear visualization of visited cells."""
        self.visited_cells.clear()

    def reserve_cells(self, path):
        """Reserves cells along the given path to prevent collisions."""
        for cell in path:
            self.reserved_cells.add(cell)

    def unreserve_cell(self, cell):
        """Clears a cell's reservation."""
        if cell in self.reserved_cells:
            self.reserved_cells.remove(cell)

    def follow_path(self, path, current_cell, target_cell):
        """Animates the movement of a cell along the given path."""

        def move_step(index, prev=None):
            if index >= len(path):
                self.activate_cell(target_cell)
                self.cells[target_cell]["active"] = True
                self.unreserve_cell(target_cell)

                # Increment filled cells counter
                self.cells_filled += 1

                # Update metrics display
                self.update_metrics_display()

                self.root.after(self.movement_speed, self.move_next_square)
                return

            cell = path[index]
            if prev:
                self.deactivate_cell(prev)
                self.unreserve_cell(prev)
            self.activate_cell(cell)
            self.root.after(self.movement_speed, lambda: move_step(index + 1, cell))

        self.deactivate_cell(current_cell)
        move_step(0, current_cell)

    def deactivate_cell(self, cell):
        """Deactivates a cell visually."""
        row, col = cell
        self.canvas.itemconfig(self.cells[(row, col)]["rect"], fill=INACTIVE_COLOR)
        self.cells[(row, col)]["active"] = False

    def activate_cell(self, cell):
        """Activates a cell visually."""
        row, col = cell
        self.canvas.itemconfig(self.cells[(row, col)]["rect"], fill=ACTIVE_COLOR)
        self.cells[(row, col)]["active"] = True

    def compare_algorithms(self):
        """Run comparison tests for all algorithms on the current configuration."""
        if self.movement_started:
            self.update_status("Cannot compare while movement is in progress.")
            return

        # Save current grid state
        saved_cells = {}
        for cell in self.cells:
            saved_cells[cell] = self.cells[cell].copy()
        saved_active_cells = self.active_cells.copy()
        saved_shape = self.target_shape.copy()

        algorithms = ["A*", "BFS", "DFS"]
        comparison_results = {}

        for algo in algorithms:
            # Reset grid to saved state
            self.reset_comparison_grid(saved_cells, saved_active_cells, saved_shape)

            # Set algorithm
            self.current_algorithm = algo
            self.algorithm_var.set(algo)

            # Run simulation
            start_time = time.time()
            success, stats = self.run_simulation_batch()
            total_time = time.time() - start_time

            comparison_results[algo] = {
                "success": success,
                "time": total_time,
                "stats": stats
            }

            # Update display
            self.update_status(f"Algorithm {algo}: {'Success' if success else 'Failed'} - {total_time:.2f}s")

        # Generate comparison report
        self.generate_comparison_report(comparison_results)

        # Reset to original state
        self.reset_comparison_grid(saved_cells, saved_active_cells, saved_shape)

    def reset_comparison_grid(self, saved_cells, saved_active_cells, saved_shape):
        """Reset grid to a saved state for comparison tests."""
        # Clear canvas
        self.canvas.delete("all")

        # Restore cells
        self.cells = {}
        for cell, properties in saved_cells.items():
            row, col = cell
            x1, y1 = col * CELL_SIZE, row * CELL_SIZE
            x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE

            fill_color = ACTIVE_COLOR if properties["active"] else INACTIVE_COLOR
            if properties["obstacle"]:
                fill_color = OBSTACLE_COLOR

            rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline=GRID_COLOR)

            self.cells[cell] = properties.copy()
            self.cells[cell]["rect"] = rect

        # Restore active cells
        self.active_cells = saved_active_cells.copy()

        # Restore target shape
        self.target_shape = saved_shape.copy()
        self.draw_green_outline()

        # Reset other state
        self.reserved_cells = set()
        self.visited_cells = set()
        self.cells_filled = 0
        self.movement_started = False

    def run_simulation_batch(self):
        """Run a complete simulation without animation for quick comparison."""
        # Set up for simulation
        self.movement_started = True
        self.cells_filled = 0

        # Stats to track
        total_path_length = 0
        total_cells_explored = 0
        num_moves = 0

        while True:
            # Check for completion or failure
            if not self.active_cells:
                return False, {
                    "path_length": total_path_length,
                    "explored": total_cells_explored,
                    "moves": num_moves
                }

            remaining_targets = [cell for cell in self.target_shape if not self.cells[cell]["active"]]
            if not remaining_targets:
                return True, {
                    "path_length": total_path_length,
                    "explored": total_cells_explored,
                    "moves": num_moves
                }

            # Calculate minimal path lengths for each target
            target_costs = {}
            target_paths = {}
            target_explored = {}

            for target in remaining_targets:
                min_length = float('inf')
                best_path = None
                best_explored = 0
                best_active = None

                for active in self.active_cells:
                    path, explored = self.find_path_batch(active, target)
                    if path and len(path) < min_length:
                        min_length = len(path)
                        best_path = path
                        best_explored = explored
                        best_active = active

                if min_length != float('inf'):
                    target_costs[target] = min_length
                    target_paths[target] = (best_path, best_active, best_explored)
                    target_explored[target] = best_explored

            if not target_costs:
                return False, {
                    "path_length": total_path_length,
                    "explored": total_cells_explored,
                    "moves": num_moves
                }

            # Select the hardest target
            hardest_target = max(target_costs, key=lambda k: target_costs[k])

            # Get the path and stats
            best_path, selected_active, explored = target_paths[hardest_target]

            # Update stats
            total_path_length += len(best_path)
            total_cells_explored += explored
            num_moves += 1

            # Move agent (without animation)
            self.active_cells.remove(selected_active)

            # Mark target as active
            self.cells[hardest_target]["active"] = True
            self.cells_filled += 1

    def find_path_batch(self, start, goal):
        """Fast pathfinding version without visualization for batch testing."""
        algorithm = self.current_algorithm

        if algorithm == "A*":
            return self.a_star_batch(start, goal)
        elif algorithm == "BFS":
            return self.bfs_search_batch(start, goal)
        elif algorithm == "DFS":
            return self.dfs_search_batch(start, goal)
        else:
            # Default to A*
            return self.a_star_batch(start, goal)

    def a_star_batch(self, start, goal):
        """A* implementation for batch testing (no visualization)."""
        explored_count = 0

        def heuristic(a, b):
            return math.dist(a, b)

        open_set = []
        heapq.heappush(open_set, (0, start))
        g_score = {start: 0}
        came_from = {}

        while open_set:
            _, current = heapq.heappop(open_set)
            explored_count += 1

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1], explored_count

            for neighbor in self.get_neighbors(current):
                if (self.cells[neighbor]["obstacle"] or
                        self.cells[neighbor]["active"] or
                        neighbor in self.reserved_cells):
                    continue

                move_cost = 1 if abs(neighbor[0] - current[0]) + abs(neighbor[1] - current[1]) == 1 else math.sqrt(2)
                tentative_g_score = g_score.get(current, float('inf')) + move_cost

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

        return None, explored_count

    def bfs_search_batch(self, start, goal):
        """BFS implementation for batch testing (no visualization)."""
        explored_count = 0

        queue = deque([[start]])
        visited = {start}

        while queue:
            path = queue.popleft()
            current = path[-1]
            explored_count += 1

            if current == goal:
                return path, explored_count

            for neighbor in self.get_neighbors(current):
                if (neighbor in visited or
                        self.cells[neighbor]["obstacle"] or
                        self.cells[neighbor]["active"] or
                        neighbor in self.reserved_cells):
                    continue

                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

        return None, explored_count

    def dfs_search_batch(self, start, goal):
        """DFS implementation for batch testing (no visualization)."""
        explored_count = 0

        stack = [[start]]
        visited = {start}

        while stack:
            path = stack.pop()
            current = path[-1]
            explored_count += 1

            if current == goal:
                return path, explored_count

            neighbors = list(self.get_neighbors(current))
            neighbors.reverse()

            for neighbor in neighbors:
                if (neighbor in visited or
                        self.cells[neighbor]["obstacle"] or
                        self.cells[neighbor]["active"] or
                        neighbor in self.reserved_cells):
                    continue

                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                stack.append(new_path)

        return None, explored_count

    def generate_comparison_report(self, results):
        """Generate a comprehensive comparison report for all algorithms."""
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)

        self.metrics_text.insert(tk.END, "=== ALGORITHM COMPARISON ===\n\n")

        for algo, data in results.items():
            success_text = "SUCCESS" if data["success"] else "FAILED"
            self.metrics_text.insert(tk.END, f"{algo}: {success_text}\n")
            self.metrics_text.insert(tk.END, f"Total Time: {data['time']:.4f} seconds\n")

            if "stats" in data:
                stats = data["stats"]
                self.metrics_text.insert(tk.END, f"Moves: {stats['moves']}\n")
                if stats['moves'] > 0:
                    self.metrics_text.insert(tk.END, f"Avg Path Length: {stats['path_length'] / stats['moves']:.2f}\n")
                    self.metrics_text.insert(tk.END, f"Avg Cells Explored: {stats['explored'] / stats['moves']:.2f}\n")

            self.metrics_text.insert(tk.END, "\n")

        # Summary comparison
        self.metrics_text.insert(tk.END, "=== SUMMARY ===\n")

        # Sort by success first, then by time
        sorted_algos = sorted(results.keys(),
                              key=lambda a: (-1 if results[a]["success"] else 0, results[a]["time"]))

        self.metrics_text.insert(tk.END, "Ranking by performance:\n")
        for i, algo in enumerate(sorted_algos):
            success_text = "✓" if results[algo]["success"] else "✗"
            self.metrics_text.insert(tk.END, f"{i + 1}. {algo} {success_text} ({results[algo]['time']:.4f}s)\n")

        self.metrics_text.config(state=tk.DISABLED)

    # Set speed for animation (can be adjusted in UI)
    @property
    def movement_speed(self):
        """Get the current movement speed."""
        return self.speed_var.get()

    @movement_speed.setter
    def movement_speed(self, value):
        """Set the movement speed."""
        self.speed_var.set(value)


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = InteractiveGrid(root)
    root.mainloop()