
import math
import tkinter as tk
import heapq

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
        self.root.title("Form a 5x4 Rectangle Shape with A* Pathfinding")
        self.canvas = tk.Canvas(root, width=GRID_SIZE * CELL_SIZE, height=GRID_SIZE * CELL_SIZE)
        self.canvas.pack()
        self.cells = {}
        self.active_cells = []
        self.reserved_cells = set()
        self.draw_grid()
        self.target_rectangle = self.define_target_rectangle()
        self.inner_cells, self.outline_cells, self.corner_cells = self.separate_cells()
        self.draw_green_outline()
        self.do_shape_button = tk.Button(root, text="Do the Shape", command=self.start_movement)
        self.do_shape_button.pack()
        self.bind_click_events()
        self.movement_started = False

    def draw_grid(self):

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x1, y1 = col * CELL_SIZE, row * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                is_active = row >= GRID_SIZE - 2
                fill_color = ACTIVE_COLOR if is_active else INACTIVE_COLOR
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline=GRID_COLOR)
                self.cells[(row, col)] = {"rect": rect, "active": is_active, "obstacle": False}
                if is_active:
                    self.active_cells.append((row, col))

    def define_target_rectangle(self):

        mid_row, mid_col = GRID_SIZE // 2, GRID_SIZE // 2
        return [(mid_row - 2 + r, mid_col - 2 + c) for r in range(4) for c in range(5)]

    def separate_cells(self):

        min_row = min(c[0] for c in self.target_rectangle)
        max_row = max(c[0] for c in self.target_rectangle)
        min_col = min(c[1] for c in self.target_rectangle)
        max_col = max(c[1] for c in self.target_rectangle)

        corners = [
            (min_row, min_col), (min_row, max_col),
            (max_row, max_col), (max_row, min_col)
        ]

        outline = []
        inner = []
        for cell in self.target_rectangle:
            if cell in corners:
                continue
            if cell[0] in (min_row, max_row) or cell[1] in (min_col, max_col):
                outline.append(cell)
            else:
                inner.append(cell)

        return inner, outline, corners

    def draw_green_outline(self):

        for cell in self.target_rectangle:
            row, col = cell
            x1, y1 = col * CELL_SIZE, row * CELL_SIZE
            x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
            rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill=OUTLINE_COLOR, outline=GRID_COLOR)
            self.cells[cell]["outline_rect"] = rect

    def bind_click_events(self):

        self.canvas.bind("<Button-1>", self.toggle_obstacle)

    def toggle_obstacle(self, event):

        if self.movement_started:
            return
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        cell = (row, col)
        if cell in self.cells:
            if self.cells[cell]["obstacle"]:
                self.canvas.itemconfig(self.cells[cell]["rect"], fill=INACTIVE_COLOR)
                self.cells[cell]["obstacle"] = False
            else:
                self.canvas.itemconfig(self.cells[cell]["rect"], fill=OBSTACLE_COLOR)
                self.cells[cell]["obstacle"] = True

    def start_movement(self):

        self.movement_started = True
        self.do_shape_button.config(state=tk.DISABLED)
        self.remove_green_outline()
        self.move_next_square()

    def remove_green_outline(self):
        for cell in self.target_rectangle:
            if "outline_rect" in self.cells[cell]:
                self.canvas.delete(self.cells[cell]["outline_rect"])
                del self.cells[cell]["outline_rect"]

    def move_next_square(self):

        if not self.active_cells:
            return

        remaining_targets = [cell for cell in self.target_rectangle if not self.cells[cell]["active"]]
        if not remaining_targets:
            return

        # Calculate minimal path lengths for each target
        target_costs = {}
        for target in remaining_targets:
            min_length = float('inf')
            for active in self.active_cells:
                path = self.a_star(active, target)
                if path:
                    min_length = min(min_length, len(path))
            if min_length != float('inf'):
                target_costs[target] = min_length

        if not target_costs:
            print("No reachable targets left")
            return

        # Select hardest target (max minimal path length)
        hardest_target = max(target_costs, key=lambda k: target_costs[k])

        # Find active cell with shortest path to hardest target
        shortest_path = None
        shortest_length = float('inf')
        selected_active = None
        for active in self.active_cells:
            path = self.a_star(active, hardest_target)
            if path and len(path) < shortest_length:
                shortest_length = len(path)
                shortest_path = path
                selected_active = active

        if not selected_active:
            self.root.after(50, self.move_next_square)
            return

        # Remove selected active cell and move
        self.active_cells.remove(selected_active)
        if shortest_path:
            self.reserve_cells(shortest_path)
            self.follow_path(shortest_path, selected_active, hardest_target)
        else:
            self.active_cells.append(selected_active)
            self.root.after(500, self.move_next_square)

    def reserve_cells(self, path):

        for cell in path:
            self.reserved_cells.add(cell)

    def unreserve_cell(self, cell):

        if cell in self.reserved_cells:
            self.reserved_cells.remove(cell)

    def a_star(self, start, goal):
        def heuristic(a, b):
            return math.dist(a, b)

        open_set = []
        heapq.heappush(open_set, (0, start))
        g_score = {start: 0}
        came_from = {}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for neighbor in self.get_neighbors(current):
                if (self.cells[neighbor]["obstacle"] or
                        self.cells[neighbor]["active"] or
                        neighbor in self.reserved_cells):
                    continue

                move_cost = 1 if abs(neighbor[0] - current[0]) + abs(neighbor[1] - current[1]) == 1 else math.sqrt(2)
                tentative_g_score = g_score.get(current, math.inf) + move_cost

                if tentative_g_score < g_score.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
        return None

    def get_neighbors(self, cell):
        
        row, col = cell
        neighbors = []
        # Cardinal moves (Up, Down, Left, Right)
        cardinal_moves = [
            (row - 1, col), (row + 1, col),
            (row, col - 1), (row, col + 1)
        ]
        for r, c in cardinal_moves:
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                if not self.cells[(r, c)]["obstacle"] and not self.cells[(r, c)]["active"]:
                    neighbors.append((r, c))

        # Diagonal moves (only allowed if both adjacent cardinal cells are free)
        diagonal_moves = [
            (row - 1, col - 1), (row - 1, col + 1),
            (row + 1, col - 1), (row + 1, col + 1)
        ]
        for i, (r, c) in enumerate(diagonal_moves):
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                if i == 0:  # top-left
                    adj1 = (row - 1, col)
                    adj2 = (row, col - 1)
                elif i == 1:  # top-right
                    adj1 = (row - 1, col)
                    adj2 = (row, col + 1)
                elif i == 2:  # bottom-left
                    adj1 = (row + 1, col)
                    adj2 = (row, col - 1)
                elif i == 3:  # bottom-right
                    adj1 = (row + 1, col)
                    adj2 = (row, col + 1)
                # Check that both adjacent cardinal cells are within bounds and free
                if (0 <= adj1[0] < GRID_SIZE and 0 <= adj1[1] < GRID_SIZE and
                    0 <= adj2[0] < GRID_SIZE and 0 <= adj2[1] < GRID_SIZE and
                    not self.cells[adj1]["obstacle"] and not self.cells[adj1]["active"] and
                    not self.cells[adj2]["obstacle"] and not self.cells[adj2]["active"]):
                    neighbors.append((r, c))
        return neighbors

    def follow_path(self, path, current_cell, target_cell):
        """Animates movement along the calculated path"""
        def move_step(index, prev=None):
            if index >= len(path):
                self.activate_cell(target_cell)
                self.cells[target_cell]["active"] = True
                self.unreserve_cell(target_cell)
                self.root.after(200, self.move_next_square)
                return

            cell = path[index]
            if prev:
                self.deactivate_cell(prev)
                self.unreserve_cell(prev)
            self.activate_cell(cell)
            self.root.after(100, lambda: move_step(index + 1, cell))

        self.deactivate_cell(current_cell)
        move_step(0, current_cell)

    def deactivate_cell(self, cell):
        row, col = cell
        self.canvas.itemconfig(self.cells[(row, col)]["rect"], fill=INACTIVE_COLOR)
        self.cells[(row, col)]["active"] = False

    def activate_cell(self, cell):
        row, col = cell
        self.canvas.itemconfig(self.cells[(row, col)]["rect"], fill=ACTIVE_COLOR)
        self.cells[(row, col)]["active"] = True


if __name__ == "__main__":
    root = tk.Tk()
    app = InteractiveGrid(root)
    root.mainloop()

