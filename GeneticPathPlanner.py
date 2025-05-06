import random
import math
import time
import numpy as np

class GeneticPathPlanner:
    """
    Genetic algorithm for optimizing paths in the grid environment.
    Works alongside existing code without modifying it.
    """
    def __init__(self, grid_app):
        self.grid_app = grid_app  # Reference to your InteractiveGrid instance
        self.population_size = 100
        self.generations = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.elitism_count = 5  # Number of best solutions to keep unchanged

    def optimize_paths(self, agent_positions, target_positions):
        """Main method to find optimized paths using genetic algorithm"""
        start_time = time.time()
        self.grid_app.update_status("Starting genetic algorithm optimization...")

        # Initialize population
        population = self.initialize_population(agent_positions, target_positions)

        # Track best solution
        best_solution = None
        best_fitness = float('-inf')

        # Evolve for several generations
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = self.evaluate_fitness(population, agent_positions, target_positions)

            # Track best solution
            gen_best_idx = fitness_scores.index(max(fitness_scores))
            gen_best_solution = population[gen_best_idx]
            gen_best_fitness = fitness_scores[gen_best_idx]

            if gen_best_fitness > best_fitness:
                best_solution = gen_best_solution
                best_fitness = gen_best_fitness

            # Select parents
            parents = self.selection(population, fitness_scores)

            # Create new population through crossover and mutation
            new_population = []

            # Elitism - keep best solutions
            sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
            for i in range(self.elitism_count):
                if i < len(sorted_indices):
                    new_population.append(population[sorted_indices[i]])

            # Fill rest of population with offspring
            while len(new_population) < self.population_size:
                # Select two parents
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)

                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation
                child1 = self.mutate(child1, agent_positions, target_positions)
                child2 = self.mutate(child2, agent_positions, target_positions)

                # Add to new population
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            # Replace old population
            population = new_population

            # Log progress
            if (generation + 1) % 5 == 0 or generation == 0:
                avg_fitness = sum(fitness_scores) / len(fitness_scores)
                self.grid_app.update_status(f"GA Generation {generation+1}/{self.generations}: "
                                          f"Best fitness = {gen_best_fitness:.2f}, "
                                          f"Avg fitness = {avg_fitness:.2f}")

        # Return best solution
        elapsed_time = time.time() - start_time
        self.grid_app.update_status(f"GA optimization completed in {elapsed_time:.2f} seconds. "
                                  f"Best fitness: {best_fitness:.2f}")

        # Decode paths from best solution
        optimized_paths = self.decode_paths(best_solution, agent_positions, target_positions)
        return optimized_paths

    def initialize_population(self, agent_positions, target_positions):
        """Create initial population of path chromosomes"""
        population = []

        # Create initial population
        for _ in range(self.population_size):
            # Each chromosome is a list of directions for each agent
            chromosome = []

            for i, (start, end) in enumerate(zip(agent_positions, target_positions)):
                # Generate a random path from start to end
                path_directions = self.generate_random_path_directions(start, end)
                chromosome.append(path_directions)

            population.append(chromosome)

        return population

    def generate_random_path_directions(self, start, end):
        """Generate a random sequence of directions to get from start to end"""
        # Direction encoding: 0=up, 1=right, 2=down, 3=left
        directions = []
        max_path_length = self.grid_app.grid_size * 3  # Limit path length

        # Sometimes use A* to get a good starting path
        if random.random() < 0.5:
            a_star_path = self.grid_app.find_path(start, end)
            if a_star_path and len(a_star_path) > 1:
                # Convert path to directions
                for i in range(len(a_star_path) - 1):
                    current = a_star_path[i]
                    next_pos = a_star_path[i + 1]

                    # Determine direction
                    dr = next_pos[0] - current[0]
                    dc = next_pos[1] - current[1]

                    if dr == -1 and dc == 0:
                        directions.append(0)  # Up
                    elif dr == 0 and dc == 1:
                        directions.append(1)  # Right
                    elif dr == 1 and dc == 0:
                        directions.append(2)  # Down
                    elif dr == 0 and dc == -1:
                        directions.append(3)  # Left

                # Add some random directions at the end for diversity
                for _ in range(random.randint(0, 10)):
                    directions.append(random.randint(0, 3))

                return directions

        # Generate random directions
        for _ in range(max_path_length):
            # Bias toward the target direction
            dr = end[0] - start[0]
            dc = end[1] - start[1]

            # Preferred directions
            preferred = []
            if dr < 0:
                preferred.append(0)  # Up
            elif dr > 0:
                preferred.append(2)  # Down

            if dc < 0:
                preferred.append(3)  # Left
            elif dc > 0:
                preferred.append(1)  # Right

            # Choose direction with bias
            if preferred and random.random() < 0.7:
                direction = random.choice(preferred)
            else:
                direction = random.randint(0, 3)

            directions.append(direction)

        return directions

    def decode_paths(self, chromosome, agent_positions, target_positions):
        """Convert direction chromosomes to actual paths"""
        paths = []

        for i, (start, target, directions) in enumerate(zip(agent_positions, target_positions, chromosome)):
            path = [start]
            current = start

            for direction in directions:
                # Direction: 0=up, 1=right, 2=down, 3=left
                dr, dc = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)][direction]
                new_r, new_c = current[0] + dr, current[1] + dc
                new_pos = (new_r, new_c)

                # Check if valid move
                if (0 <= new_r < self.grid_app.grid_size and
                    0 <= new_c < self.grid_app.grid_size and
                    not self.grid_app.cells.get(new_pos, {}).get("obstacle", False)):
                    path.append(new_pos)
                    current = new_pos

                # Stop if target reached
                if current == target:
                    break

            # If path doesn't reach target, try to find a direct path
            if path[-1] != target:
                direct_path = self.grid_app.find_path(path[-1], target)
                if direct_path and len(direct_path) > 1:
                    path.extend(direct_path[1:])

            paths.append(path)

        return paths

    def evaluate_fitness(self, population, agent_positions, target_positions):
        """Evaluate fitness of each chromosome in the population"""
        fitness_scores = []

        # Identify shape border if not already done
        if not hasattr(self, 'shape_border'):
            self.shape_border = self.identify_shape_border(target_positions)

        for chromosome in population:
            # Decode paths
            paths = self.decode_paths(chromosome, agent_positions, target_positions)

            # Calculate fitness based on multiple factors
            path_lengths = [len(path) for path in paths]
            targets_reached = sum(1 for i, path in enumerate(paths)
                                if path and path[-1] == target_positions[i])

            # Count collisions
            collisions = self.count_collisions(paths)

            # Calculate path efficiency (how direct are the paths)
            efficiency = 0
            for i, path in enumerate(paths):
                if path:
                    # Ideal path length (Manhattan distance)
                    ideal_length = abs(agent_positions[i][0] - target_positions[i][0]) + \
                                  abs(agent_positions[i][1] - target_positions[i][1])
                    # Actual path length
                    actual_length = len(path) - 1
                    # Efficiency ratio (1.0 is perfect)
                    if actual_length > 0:
                        path_efficiency = ideal_length / actual_length
                        efficiency += path_efficiency

            # Normalize efficiency
            if paths:
                efficiency /= len(paths)

            # Calculate shape formation quality
            shape_formation_quality = self.calculate_shape_formation_quality(paths, target_positions)

            # Calculate border reaching score
            border_score = self.calculate_border_reaching_score(paths)

            # Calculate collision density within shape
            shape_collisions = self.count_shape_collisions(paths, target_positions)

            # Fitness formula: prioritize reaching targets, minimize path length, avoid collisions
            # Significantly increase collision penalty, especially within shape
            fitness = (targets_reached * 1000) + \
                     (efficiency * 500) - \
                     (sum(path_lengths) * 0.1) - \
                     (collisions * 300) - \
                     (shape_collisions * 500) + \
                     (border_score * 200) + \
                     (shape_formation_quality * 800)

            fitness_scores.append(fitness)

        return fitness_scores

    def count_collisions(self, paths):
        """Count collisions between agent paths"""
        collisions = 0

        # Check for position collisions (two agents in same cell at same time)
        all_positions = {}
        for agent_idx, path in enumerate(paths):
            for time_step, pos in enumerate(path):
                if (time_step, pos) in all_positions:
                    collisions += 1
                all_positions[(time_step, pos)] = agent_idx

        # Check for crossing collisions (agents swap positions)
        for agent1 in range(len(paths)):
            path1 = paths[agent1]
            for agent2 in range(agent1 + 1, len(paths)):
                path2 = paths[agent2]
                for t in range(min(len(path1), len(path2)) - 1):
                    if t+1 < len(path1) and t+1 < len(path2):
                        if path1[t] == path2[t+1] and path1[t+1] == path2[t]:
                            collisions += 1

        return collisions

    def identify_shape_border(self, target_positions):
        """Identify the border cells of the target shape"""
        if not target_positions:
            return []

        # Find min/max coordinates to determine shape boundaries
        min_row = min(pos[0] for pos in target_positions)
        max_row = max(pos[0] for pos in target_positions)
        min_col = min(pos[1] for pos in target_positions)
        max_col = max(pos[1] for pos in target_positions)

        # Create a set of all target positions for faster lookup
        target_set = set(target_positions)

        # Identify border cells (cells that have at least one non-shape neighbor)
        border_cells = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right

        for pos in target_positions:
            # Check if this cell has any neighbors that are not part of the shape
            for dr, dc in directions:
                neighbor = (pos[0] + dr, pos[1] + dc)

                # If neighbor is outside the grid or not in the shape, this is a border cell
                if (neighbor not in target_set and
                    0 <= neighbor[0] < self.grid_app.grid_size and
                    0 <= neighbor[1] < self.grid_app.grid_size):
                    border_cells.append(pos)
                    break

        return border_cells

    def calculate_border_reaching_score(self, paths):
        """Calculate a score based on how many paths reach the shape border"""
        if not hasattr(self, 'shape_border') or not self.shape_border:
            return 0

        # Convert border to a set for faster lookup
        border_set = set(self.shape_border)

        # Count how many paths reach the border
        border_reached_count = 0
        for path in paths:
            for pos in path:
                if pos in border_set:
                    border_reached_count += 1
                    break

        # Normalize by number of paths
        if paths:
            return border_reached_count / len(paths)
        return 0

    def count_shape_collisions(self, paths, target_positions):
        """Count collisions that occur within the target shape area"""
        if not target_positions:
            return 0

        # Create a set of all target positions for faster lookup
        target_set = set(target_positions)

        # Count collisions within the shape
        shape_collisions = 0
        all_positions = {}

        for agent_idx, path in enumerate(paths):
            for time_step, pos in enumerate(path):
                # Check if this position is within the shape
                if pos in target_set:
                    # Check for collision at this position and time
                    if (time_step, pos) in all_positions:
                        shape_collisions += 1
                    all_positions[(time_step, pos)] = agent_idx

        return shape_collisions

    def calculate_shape_formation_quality(self, paths, target_positions):
        """Calculate how well the final positions of agents match the target shape"""
        if not paths or not target_positions:
            return 0

        # Get final positions of all agents
        final_positions = [path[-1] if path else None for path in paths]

        # Count how many agents reached their exact target
        exact_matches = sum(1 for i, pos in enumerate(final_positions)
                          if pos is not None and i < len(target_positions) and pos == target_positions[i])

        # Calculate the total distance from final positions to targets
        total_distance = 0
        for i, pos in enumerate(final_positions):
            if pos is not None and i < len(target_positions):
                target = target_positions[i]
                distance = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
                total_distance += distance

        # Normalize distances
        max_possible_distance = self.grid_app.grid_size * 2 * len(paths)
        distance_score = 1.0 - (total_distance / max_possible_distance if max_possible_distance > 0 else 0)

        # Combine exact matches and distance score
        # Weight exact matches more heavily
        quality = (0.7 * exact_matches / len(paths) if paths else 0) + (0.3 * distance_score)

        return quality

    def selection(self, population, fitness_scores):
        """Select parents using tournament selection"""
        parents = []

        # Tournament selection
        tournament_size = 5
        for _ in range(self.population_size):
            tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            parents.append(population[winner_idx])

        return parents

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents"""
        child1, child2 = [], []

        # Crossover each agent's path separately
        for p1_directions, p2_directions in zip(parent1, parent2):
            # Single point crossover
            if p1_directions and p2_directions:
                point = random.randint(1, min(len(p1_directions), len(p2_directions)))
                c1_directions = p1_directions[:point] + p2_directions[point:]
                c2_directions = p2_directions[:point] + p1_directions[point:]
            else:
                c1_directions, c2_directions = p1_directions.copy(), p2_directions.copy()

            child1.append(c1_directions)
            child2.append(c2_directions)

        return child1, child2

    def mutate(self, chromosome, agent_positions, target_positions):
        """Apply mutation to a chromosome"""
        mutated = []

        for agent_idx, directions in enumerate(chromosome):
            if random.random() < self.mutation_rate and directions:
                # Choose mutation type
                mutation_type = random.choice(['add', 'remove', 'change'])

                if mutation_type == 'add' and len(directions) < 100:
                    # Add a random direction
                    insert_pos = random.randint(0, len(directions))
                    new_direction = random.randint(0, 3)
                    new_directions = directions[:insert_pos] + [new_direction] + directions[insert_pos:]

                elif mutation_type == 'remove' and len(directions) > 1:
                    # Remove a random direction
                    remove_pos = random.randint(0, len(directions) - 1)
                    new_directions = directions[:remove_pos] + directions[remove_pos + 1:]

                else:  # change
                    # Change a random direction
                    if directions:
                        change_pos = random.randint(0, len(directions) - 1)
                        new_direction = random.randint(0, 3)
                        new_directions = directions.copy()
                        new_directions[change_pos] = new_direction
                    else:
                        new_directions = directions

                mutated.append(new_directions)
            else:
                mutated.append(directions)

        return mutated
