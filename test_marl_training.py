"""
Simple test script to demonstrate MARL training progress in the terminal.
Run this script to see the training progress printed to your terminal.
"""

from PathPlanningMARL import PathPlanningMARL

def main():
    # Create a sample grid with obstacles
    grid_size = 10
    grid_state = {}
    
    # Initialize all cells as empty
    for row in range(grid_size):
        for col in range(grid_size):
            grid_state[(row, col)] = {
                "obstacle": False,
                "active": False
            }
    
    # Add some obstacles
    for i in range(3, 7):
        grid_state[(i, 4)]["obstacle"] = True
    
    for i in range(2, 5):
        grid_state[(7, i)]["obstacle"] = True
    
    # Initialize MARL system with a small number of agents
    marl = PathPlanningMARL(
        grid_size=grid_size, 
        num_agents=3,
        epsilon=1.0,  # Start with full exploration
        epsilon_decay=0.9,  # Faster decay for this demo
        memory_size=1000
    )
    
    # Train for a small number of episodes to see progress quickly
    marl.train_agents(
        grid_state=grid_state,
        num_episodes=20,  # Small number for quick demonstration
        max_steps=50,
        save_interval=10,  # Save every 10 episodes
        print_interval=2   # Print progress every 2 episodes
    )
    
    # Test the trained agents
    print("\nTesting trained agents...")
    
    # Set initial positions and targets
    agent_positions = [(0, 0), (0, 9), (9, 0)]
    target_positions = [(9, 9), (9, 5), (5, 9)]
    
    # Plan paths using the trained models
    result = marl.plan_paths(
        grid_state, 
        agent_positions, 
        target_positions, 
        max_steps=50, 
        training=False
    )
    
    if result:
        # Visualize the paths
        print("\nPath visualization:")
        print(marl.visualize_paths(grid_state, result['paths'], agent_positions, target_positions))
        
        # Print results
        print("\nTest results:")
        for i in range(len(agent_positions)):
            print(f"Agent {i+1} - Success: {result['success'][i]}, Reward: {result['rewards'][i]:.2f}")
            print(f"Path: {result['paths'][i]}")
            print()

if __name__ == "__main__":
    main()
