from MARL_Integration_For_Parallel import integrate_marl_with_app
def start_movement(self):
    """Start the movement of cells to form the target shape."""
    if self.movement_started:
        return

    self.movement_started = True
    self.do_shape_btn.config(state=tk.DISABLED)

    self.start_time = time.time()
    self.cells_filled = 0

    self.metrics[self.current_algorithm]["total_targets"] = len(self.target_shape)
    self.metrics[self.current_algorithm]["completed_targets"] = 0
    if hasattr(self, 'marl_integration') and self.parallel_mode and self.marl_integration.marl is not None:
        self.update_status("Using MARL for parallel movement")
        self.marl_integration.do_shape_with_marl()
        return

    if self.parallel_centralized_mode:
        self.update_status("Starting parallel centralized movement (F1 safety car queue)")
        self.start_parallel_centralized_movement()
    elif self.parallel_mode:
        self.update_status("Starting parallel movement")
        self.movement_timer = self.root.after(50, self.process_parallel_movements)
    else:
        self.update_status(f"Starting sequential movement with {self.current_algorithm} algorithm.")
        self.move_next_square()
