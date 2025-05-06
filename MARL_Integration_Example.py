"""
Example of how to integrate MARL with your existing application.
Add this code to your import math.py file to enable MARL training.
"""

# Add this import at the top of your file
from MARL_Integration_For_Parallel import integrate_marl_with_app

# Add this line at the end of your __init__ method in InteractiveGrid class
# self.marl_integration = integrate_marl_with_app(self)

# Example of how to modify your start_movement method to use MARL if available
def start_movement(self):
    """Start the movement of cells to form the target shape."""
    if self.movement_started:
        return
    
    self.movement_started = True
    self.do_shape_btn.config(state=tk.DISABLED)
    
    # Reset metrics
    self.start_time = time.time()
    self.cells_filled = 0
    
    # Update metrics for the current algorithm
    self.metrics[self.current_algorithm]["total_targets"] = len(self.target_shape)
    self.metrics[self.current_algorithm]["completed_targets"] = 0
    
    # Check if we should use MARL for parallel movement
    if hasattr(self, 'marl_integration') and self.parallel_mode and self.marl_integration.marl is not None:
        # Use MARL for parallel movement
        self.update_status("Using MARL for parallel movement")
        self.marl_integration.do_shape_with_marl()
        return
    
    # Original movement code
    if self.parallel_centralized_mode:
        # Start parallel centralized movement (F1 safety car queue)
        self.update_status("Starting parallel centralized movement (F1 safety car queue)")
        self.start_parallel_centralized_movement()
    elif self.parallel_mode:
        # Start parallel movement
        self.update_status("Starting parallel movement")
        self.movement_timer = self.root.after(50, self.process_parallel_movements)
    else:
        # Original sequential movement
        self.update_status(f"Starting sequential movement with {self.current_algorithm} algorithm.")
        self.move_next_square()
