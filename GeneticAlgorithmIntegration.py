import tkinter as tk
from GeneticPathPlanner import GeneticPathPlanner

def integrate_ga_with_app(app):
    """
    Integrate Genetic Algorithm with the InteractiveGrid application.

    Args:
        app: The InteractiveGrid application instance

    Returns:
        GeneticPathPlanner instance
    """
    ga_header = tk.Frame(app.root, bg="#34495E", padx=5, pady=2)
    ga_header.pack(fill=tk.X, pady=(10, 0))

    tk.Label(ga_header, text="Genetic Algorithm Path Planning",
            font=("Arial", 10, "bold"), bg="#34495E", fg="white").pack(anchor=tk.W)

    ga_frame = tk.Frame(app.root, bg="#F8F9FA")
    ga_frame.pack(pady=5, fill=tk.X)

    settings_frame = tk.Frame(ga_frame, bg="#F8F9FA")
    settings_frame.pack(pady=5, fill=tk.X)

    pop_frame = tk.Frame(settings_frame, bg="#F8F9FA")
    pop_frame.pack(side=tk.LEFT, padx=10)

    tk.Label(pop_frame, text="Population:", bg="#F8F9FA").pack(side=tk.LEFT)
    pop_var = tk.IntVar(value=100)
    pop_entry = tk.Entry(pop_frame, textvariable=pop_var, width=5)
    pop_entry.pack(side=tk.LEFT, padx=5)

    gen_frame = tk.Frame(settings_frame, bg="#F8F9FA")
    gen_frame.pack(side=tk.LEFT, padx=10)

    tk.Label(gen_frame, text="Generations:", bg="#F8F9FA").pack(side=tk.LEFT)
    gen_var = tk.IntVar(value=50)
    gen_entry = tk.Entry(gen_frame, textvariable=gen_var, width=5)
    gen_entry.pack(side=tk.LEFT, padx=5)

    button_style = {
        "bg": "#9B59B6",
        "fg": "white",
        "activebackground": "#8E44AD",
        "relief": tk.RAISED,
        "padx": 10,
        "pady": 5
    }

    buttons_frame = tk.Frame(ga_frame, bg="#F8F9FA")
    buttons_frame.pack(pady=5, fill=tk.X)

    ga_button = tk.Button(
        buttons_frame,
        text="Plan Paths with GA",
        command=lambda: run_genetic_algorithm(app, pop_var.get(), gen_var.get()),
        **button_style
    )
    ga_button.pack(side=tk.LEFT, padx=5)

    ga_execute_button = tk.Button(
        buttons_frame,
        text="Do Shape with GA",
        command=lambda: do_shape_with_ga(app, pop_var.get(), gen_var.get()),
        **button_style
    )
    ga_execute_button.pack(side=tk.LEFT, padx=5)

    app.ga_planner = GeneticPathPlanner(app)
    app.ga_ui = {
        'pop_var': pop_var,
        'gen_var': gen_var,
        'ga_button': ga_button,
        'ga_execute_button': ga_execute_button
    }

    return app.ga_planner

def run_genetic_algorithm(app, population_size=100, generations=50):
    """
    Run the genetic algorithm to find optimal paths.

    Args:
        app: The InteractiveGrid application instance
        population_size: Size of the GA population
        generations: Number of generations to evolve

    Returns:
        List of optimized paths
    """
    if app.movement_started:
        app.update_status("Cannot start GA while movement is in progress")
        return None

    app.ga_planner.population_size = population_size
    app.ga_planner.generations = generations

    agent_positions = app.active_cells.copy()
    target_positions = app.target_shape[:len(agent_positions)]

    if not agent_positions or not target_positions:
        app.update_status("No active cells or target shape defined")
        return None

    app.update_status(f"Running genetic algorithm with population={population_size}, generations={generations}...")
    optimized_paths = app.ga_planner.optimize_paths(agent_positions, target_positions)

    visualize_paths(app, agent_positions, target_positions, optimized_paths)

    app.update_status(f"GA optimization complete. Found paths for {len(optimized_paths)} agents.")

    return optimized_paths

def do_shape_with_ga(app, population_size=100, generations=50):
    """
    Run the genetic algorithm and then execute the movement.

    Args:
        app: The InteractiveGrid application instance
        population_size: Size of the GA population
        generations: Number of generations to evolve
    """
    optimized_paths = run_genetic_algorithm(app, population_size, generations)

    if not optimized_paths:
        return

    agent_positions = app.active_cells.copy()
    target_positions = app.target_shape[:len(agent_positions)]

    app.update_status("Starting movement with GA-optimized paths")
    start_movement_with_paths(app, agent_positions, target_positions, optimized_paths)

def start_movement_with_paths(app, agent_positions, target_positions, paths):
    """
    Start movement using GA-generated paths.

    Args:
        app: The InteractiveGrid application instance
        agent_positions: List of agent positions
        target_positions: List of target positions
        paths: List of paths for each agent
    """
    app.movement_started = True
    app.do_shape_btn.config(state=tk.DISABLED)

    if hasattr(app, 'ga_ui'):
        app.ga_ui['ga_button'].config(state=tk.DISABLED)
        app.ga_ui['ga_execute_button'].config(state=tk.DISABLED)

    app.moving_cells = {}

    if not hasattr(app, 'ga_shape_border'):
        app.ga_shape_border = identify_shape_border(app, target_positions)
        app.update_status(f"Identified {len(app.ga_shape_border)} border cells in the target shape")

    for i, (agent_pos, target_pos, path) in enumerate(zip(agent_positions, target_positions, paths)):
        if not path:
            continue

        move_id = f"ga_{i}"

        app.moving_cells[move_id] = (path, 1, target_pos, agent_pos, False)

        if agent_pos in app.active_cells:
            app.active_cells.remove(agent_pos)

    print("DEBUG: Replacing standard update_moving_cells with custom version")
    app.original_update_moving_cells = app.update_moving_cells
    app.update_moving_cells = custom_update_moving_cells

    app.movement_timer = app.root.after(app.movement_speed, app.update_moving_cells)

def identify_shape_border(app, target_positions):
    """
    Identify the border cells of the target shape.

    Args:
        app: The InteractiveGrid application instance
        target_positions: List of target positions

    Returns:
        List of border cells
    """
    if not target_positions:
        return []

    target_set = set(target_positions)

    border_cells = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for pos in target_positions:
        for dr, dc in directions:
            neighbor = (pos[0] + dr, pos[1] + dc)

            if (neighbor not in target_set and
                0 <= neighbor[0] < app.grid_size and
                0 <= neighbor[1] < app.grid_size):
                border_cells.append(pos)
                break

    return border_cells

def custom_update_moving_cells(app):
    """
    Custom implementation of update_moving_cells that handles shape border detection
    and allows cells to move freely once they're within the shape border.

    Args:
        app: The InteractiveGrid application instance
    """
    print("DEBUG: Custom update_moving_cells called")

    completed_moves = []
    next_positions = {}
    collision_cells = set()

    shape_border_set = set(app.ga_shape_border) if hasattr(app, 'ga_shape_border') else set()
    print(f"DEBUG: Shape border has {len(shape_border_set)} cells")

    for move_id, move_data in list(app.moving_cells.items()):
        if len(move_data) < 5:
            path, index, target, start_cell = move_data
            app.moving_cells[move_id] = (path, index, target, start_cell, False)
            in_shape_border = False
        else:
            path, index, target, start_cell, in_shape_border = move_data

        if index >= len(path):
            completed_moves.append(move_id)
            continue

        current_pos = path[index - 1] if index > 0 else start_cell
        next_pos = path[index]

        if not in_shape_border and next_pos in shape_border_set:
            print(f"DEBUG: Cell {move_id} reached shape border at {next_pos}")
            app.update_status(f"Cell {move_id} reached shape border at {next_pos}")
            app.moving_cells[move_id] = (path, index, target, start_cell, True)
            in_shape_border = True

        if in_shape_border:
            if current_pos != target:
                direct_path = app.find_path(current_pos, target)
                if direct_path and len(direct_path) > 1:
                    new_path = [current_pos] + direct_path[1:]
                    app.moving_cells[move_id] = (new_path, 1, target, current_pos, True)
                    next_pos = new_path[1]
                    print(f"DEBUG: Recalculated path for cell {move_id} within shape border")

        if next_pos in next_positions:
            collision_cells.add(next_pos)
        else:
            next_positions[next_pos] = move_id

    print(f"DEBUG: Detected {len(collision_cells)} collision cells")
    moves_executed = 0

    for move_id in list(app.moving_cells.keys()):
        if move_id in completed_moves:
            del app.moving_cells[move_id]
            continue

        move_data = app.moving_cells[move_id]
        if len(move_data) < 5:
            path, index, target, start_cell = move_data
            in_shape_border = False
        else:
            path, index, target, start_cell, in_shape_border = move_data

        if index >= len(path):
            del app.moving_cells[move_id]
            continue

        next_pos = path[index]
        current_pos = path[index - 1] if index > 0 else start_cell

        if next_pos in collision_cells:
            print(f"DEBUG: Skipping move for cell {move_id} due to collision at {next_pos}")
            continue

        if next_pos in next_positions and next_positions[next_pos] == move_id:
            try:
                if hasattr(app, 'deactivate_cell'):
                    app.deactivate_cell(current_pos)
                else:
                    app.cells[current_pos]["active"] = False
                    app.canvas.itemconfig(app.cells[current_pos]["rect"], fill=app.INACTIVE_COLOR)

                if hasattr(app, 'activate_cell'):
                    app.activate_cell(next_pos)
                else:
                    app.cells[next_pos]["active"] = True
                    app.canvas.itemconfig(app.cells[next_pos]["rect"], fill=app.ACTIVE_COLOR)

                app.moving_cells[move_id] = (path, index + 1, target, start_cell, in_shape_border)
                moves_executed += 1

                if next_pos == target:
                    completed_moves.append(move_id)
                    print(f"DEBUG: Cell {move_id} reached target {target}")
            except Exception as e:
                print(f"DEBUG: Error moving cell {move_id}: {str(e)}")
                app.update_status(f"Error moving cell {move_id}: {str(e)}")

    print(f"DEBUG: Executed {moves_executed} moves this step")

    for move_id in completed_moves:
        if move_id in app.moving_cells:
            move_data = app.moving_cells[move_id]
            if len(move_data) < 5:
                path, index, target, start_cell = move_data
            else:
                path, index, target, start_cell, in_shape_border = move_data

            if hasattr(app, 'activate_cell'):
                app.activate_cell(target)
            else:
                app.cells[target]["active"] = True
                app.canvas.itemconfig(app.cells[target]["rect"], fill=app.ACTIVE_COLOR)

            del app.moving_cells[move_id]
            print(f"DEBUG: Completed movement for cell {move_id}")

    if not app.moving_cells:
        app.update_status("All movements complete")
        app.movement_started = False
        print("DEBUG: All movements complete")

        app.do_shape_btn.config(state=tk.NORMAL)
        if hasattr(app, 'ga_ui'):
            app.ga_ui['ga_button'].config(state=tk.NORMAL)
            app.ga_ui['ga_execute_button'].config(state=tk.NORMAL)

        if hasattr(app, 'original_update_moving_cells'):
            app.update_moving_cells = app.original_update_moving_cells
            print("DEBUG: Restored original update_moving_cells function")

        return

    app.movement_timer = app.root.after(app.movement_speed, app.update_moving_cells)

def visualize_paths(app, agent_positions, target_positions, paths):
    """
    Visualize the paths on the grid without actually moving agents.

    Args:
        app: The InteractiveGrid application instance
        agent_positions: List of agent positions
        target_positions: List of target positions
        paths: List of paths for each agent
    """
    app.canvas.delete("path_preview")

    colors = ["#E74C3C", "#2ECC71", "#3498DB", "#F39C12", "#9B59B6",
              "#1ABC9C", "#D35400", "#34495E", "#16A085", "#27AE60"]

    for i, path in enumerate(paths):
        if len(path) < 2:
            continue

        color = colors[i % len(colors)]

        for j in range(len(path) - 1):
            start_pos = path[j]
            end_pos = path[j + 1]

            start_x = start_pos[1] * app.cell_size + app.cell_size // 2
            start_y = start_pos[0] * app.cell_size + app.cell_size // 2
            end_x = end_pos[1] * app.cell_size + app.cell_size // 2
            end_y = end_pos[0] * app.cell_size + app.cell_size // 2

            app.canvas.create_line(
                start_x, start_y, end_x, end_y,
                fill=color, width=2, arrow=tk.LAST,
                tags="path_preview"
            )

    for i, pos in enumerate(agent_positions):
        x = pos[1] * app.cell_size + app.cell_size // 2
        y = pos[0] * app.cell_size + app.cell_size // 2

        app.canvas.create_text(
            x, y, text=str(i+1),
            fill="black", font=("Arial", 9, "bold"),
            tags="path_preview"
        )

    if hasattr(app, 'ga_planner') and hasattr(app.ga_planner, 'shape_border'):
        for pos in app.ga_planner.shape_border:
            x = pos[1] * app.cell_size + app.cell_size // 2
            y = pos[0] * app.cell_size + app.cell_size // 2

            app.canvas.create_oval(
                x - 5, y - 5, x + 5, y + 5,
                outline="#FF5733", width=2,
                tags="path_preview"
            )

    app.update_status("Path preview: Green lines show GA-optimized paths")
