import tkinter as tk
from tkinter import ttk
import threading
import os
import sys
import traceback
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('MARL_Integration')

# Import the MARL module
try:
    from PathPlanningMARL import PathPlanningMARL
    logger.info("Successfully imported PathPlanningMARL")
except ImportError as e:
    logger.error(f"Failed to import PathPlanningMARL: {e}")
    traceback.print_exc()

def integrate_marl_with_app(app):
    """
    Simple integration of MARL with the InteractiveGrid application.

    Args:
        app: The InteractiveGrid application instance
    """
    logger.info(f"Integrating MARL with app: {app}")

    try:
        # Check if app has the required attributes
        if not hasattr(app, 'root'):
            logger.error("App does not have 'root' attribute")
            return None

        # Check if app already has a MARL integration
        if hasattr(app, 'marl_integration') and app.marl_integration is not None:
            # If it's already a dictionary from this module, return it
            if isinstance(app.marl_integration, dict):
                logger.info("App already has a dictionary MARL integration, returning existing integration")
                return app.marl_integration
            # If it's an object from another module, don't override it
            elif hasattr(app.marl_integration, 'marl'):
                logger.warning("App already has an object MARL integration from another module")
                logger.warning("Not overriding existing MARL integration")
                return app.marl_integration

        # Create a frame at the bottom of the window
        marl_frame = tk.Frame(app.root, bg="#34495E", padx=10, pady=5)
        marl_frame.pack(fill=tk.X, side=tk.BOTTOM)
        logger.debug("Created MARL frame")

        # Add a label
        tk.Label(
            marl_frame,
            text="Multi-Agent Reinforcement Learning",
            font=("Arial", 10, "bold"),
            bg="#34495E",
            fg="white"
        ).pack(side=tk.LEFT, padx=5)

        # Add episodes input
        tk.Label(
            marl_frame,
            text="Episodes:",
            bg="#34495E",
            fg="white"
        ).pack(side=tk.LEFT, padx=5)

        episodes_var = tk.IntVar(value=20)
        episodes_entry = tk.Entry(marl_frame, textvariable=episodes_var, width=5)
        episodes_entry.pack(side=tk.LEFT, padx=5)

        # Add train button
        train_btn = tk.Button(
            marl_frame,
            text="Train MARL for Parallel Movement",
            command=lambda: start_marl_training(app, episodes_var.get()),
            bg="#2ECC71",
            fg="white",
            activebackground="#27AE60"
        )
        train_btn.pack(side=tk.LEFT, padx=10)
        logger.debug("Created train button")

        # Add use MARL button
        use_marl_btn = tk.Button(
            marl_frame,
            text="Do Shape with MARL",
            command=lambda: do_shape_with_marl(app),
            bg="#3498DB",
            fg="white",
            activebackground="#2980B9",
            state=tk.DISABLED
        )
        use_marl_btn.pack(side=tk.LEFT, padx=10)

        # Add progress bar
        progress_bar = ttk.Progressbar(marl_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        progress_bar.pack(side=tk.LEFT, padx=10)

        # Add status label
        status_label = tk.Label(marl_frame, text="Not started", bg="#34495E", fg="white")
        status_label.pack(side=tk.LEFT, padx=5)

        # Store references
        marl_integration = {
            'marl': None,
            'episodes_var': episodes_var,
            'train_btn': train_btn,
            'use_marl_btn': use_marl_btn,
            'progress_bar': progress_bar,
            'status_label': status_label,
            'training_active': False,
            'training_progress': 0
        }

        # Set the marl_integration attribute on the app
        app.marl_integration = marl_integration
        logger.info("Successfully created MARL integration")

        return app.marl_integration

    except Exception as e:
        logger.error(f"Error in integrate_marl_with_app: {e}")
        traceback.print_exc()
        return None

def start_marl_training(app, episodes):
    """Start MARL training."""
    logger.info(f"Starting MARL training with {episodes} episodes")

    try:
        # Check if app has the required attributes
        if not hasattr(app, 'marl_integration'):
            logger.error("App does not have 'marl_integration' attribute")
            if hasattr(app, 'update_status'):
                app.update_status("Error: MARL integration not initialized")
            return

        # Check if app.marl_integration is a dictionary (from this module)
        if not isinstance(app.marl_integration, dict):
            logger.error("App.marl_integration is not a dictionary, it might be from another module")
            if hasattr(app, 'update_status'):
                app.update_status("Error: MARL integration is not from Simple_MARL_Integration")
            return

        # Check if training is already active
        if 'training_active' in app.marl_integration and app.marl_integration['training_active']:
            logger.warning("Training already in progress")
            app.update_status("Training already in progress!")
            return

        # Check if app has the required attributes for grid state
        if not hasattr(app, 'cells'):
            logger.error("App does not have 'cells' attribute")
            app.update_status("Error: App does not have cells attribute")
            return

        # Check if app has active cells
        if not hasattr(app, 'active_cells'):
            logger.error("App does not have 'active_cells' attribute")
            app.update_status("Error: App does not have active_cells attribute")
            return

        # Check if app has grid size
        if not hasattr(app, 'grid_size'):
            logger.error("App does not have 'grid_size' attribute")
            app.update_status("Error: App does not have grid_size attribute")
            return

        # Convert grid to MARL format
        logger.debug("Converting grid to MARL format")
        grid_state = {}
        for cell in app.cells:
            grid_state[cell] = {
                "obstacle": app.cells[cell]["obstacle"],
                "active": app.cells[cell]["active"]
            }
        logger.debug(f"Grid state created with {len(grid_state)} cells")

        # Initialize MARL
        logger.info(f"Initializing MARL with grid_size={app.grid_size}, num_agents={len(app.active_cells)}")
        try:
            app.marl_integration['marl'] = PathPlanningMARL(
                grid_size=app.grid_size,
                num_agents=len(app.active_cells)
            )
            logger.info("MARL initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MARL: {e}")
            app.update_status(f"Error initializing MARL: {str(e)}")
            traceback.print_exc()
            return

        # Reset progress
        logger.debug("Resetting progress")
        app.marl_integration['training_progress'] = 0

        # Update UI elements safely
        try:
            if 'progress_bar' in app.marl_integration:
                app.marl_integration['progress_bar']['value'] = 0

            if 'status_label' in app.marl_integration:
                app.marl_integration['status_label'].config(text="Starting...")

            # Disable buttons
            if 'train_btn' in app.marl_integration:
                app.marl_integration['train_btn'].config(state=tk.DISABLED)

            if 'use_marl_btn' in app.marl_integration:
                app.marl_integration['use_marl_btn'].config(state=tk.DISABLED)

            if hasattr(app, 'do_shape_btn'):
                app.do_shape_btn.config(state=tk.DISABLED)
        except Exception as e:
            logger.error(f"Error updating UI: {e}")
            # Continue anyway

        # Start training in a separate thread
        logger.info("Starting training thread")
        app.marl_integration['training_active'] = True
        training_thread = threading.Thread(
            target=run_marl_training,
            args=(app, grid_state, episodes)
        )
        training_thread.daemon = True
        training_thread.start()
        logger.debug("Training thread started")

        # Start progress update
        logger.debug("Starting progress update")
        update_training_progress(app)

        app.update_status("MARL training started for parallel movement...")
        logger.info("MARL training started successfully")

    except Exception as e:
        logger.error(f"Error in start_marl_training: {e}")
        traceback.print_exc()
        if hasattr(app, 'update_status'):
            app.update_status(f"Error starting MARL training: {str(e)}")
        # Try to reset training state
        try:
            if hasattr(app, 'marl_integration') and isinstance(app.marl_integration, dict):
                app.marl_integration['training_active'] = False
        except:
            pass

def run_marl_training(app, grid_state, episodes):
    """Run MARL training in a separate thread."""
    logger.info(f"Running MARL training with {episodes} episodes")

    try:
        # First check if MARL integration is properly initialized
        if not hasattr(app, 'marl_integration') or app.marl_integration is None:
            logger.error("MARL integration not initialized")
            if hasattr(app, 'update_status'):
                app.update_status("Error: MARL integration not initialized")
            return

        # Check if app.marl_integration is a dictionary (from this module)
        if not isinstance(app.marl_integration, dict):
            logger.error("App.marl_integration is not a dictionary, it might be from another module")
            if hasattr(app, 'update_status'):
                app.update_status("Error: MARL integration is not from Simple_MARL_Integration")
            return

        # Check if app has root for UI updates
        has_root = hasattr(app, 'root')
        if not has_root:
            logger.warning("App does not have 'root' attribute, UI updates will be skipped")

        # Check if GPU is available and being used
        try:
            import tensorflow as tf
            logger.info("Checking for GPU availability")
            physical_devices = tf.config.list_physical_devices()
            device_names = [device.name for device in physical_devices]
            logger.info(f"Available devices: {device_names}")

            if hasattr(app, 'update_status'):
                app.update_status(f"Available devices: {device_names}")

            # Update UI with GPU status
            gpu_devices = tf.config.list_physical_devices('GPU')
            if gpu_devices:
                gpu_info = gpu_devices[0].name
                logger.info(f"Training will use GPU: {gpu_info}")

                if hasattr(app, 'update_status'):
                    app.update_status(f"Training will use GPU: {gpu_info}")

                # Safely update status label
                if has_root and 'status_label' in app.marl_integration:
                    app.root.after(0, lambda: app.marl_integration['status_label'].config(text=f"Using GPU: {gpu_info}"))
            else:
                logger.info("No GPU available, training will use CPU")

                if hasattr(app, 'update_status'):
                    app.update_status("Training will use CPU (no GPU available)")

                # Safely update status label
                if has_root and 'status_label' in app.marl_integration:
                    app.root.after(0, lambda: app.marl_integration['status_label'].config(text="Using CPU for training"))
        except Exception as e:
            logger.error(f"Error checking GPU: {e}")
            traceback.print_exc()

            if hasattr(app, 'update_status'):
                app.update_status(f"Error checking GPU: {str(e)}")

        # Check if MARL model is initialized
        if 'marl' not in app.marl_integration or app.marl_integration['marl'] is None:
            logger.error("MARL model not initialized")

            if hasattr(app, 'update_status'):
                app.update_status("Error: MARL model not initialized")

            if has_root:
                app.root.after(0, lambda: training_completed(app, False))
            return

        # Train the MARL system
        logger.info("Starting MARL training...")

        if hasattr(app, 'update_status'):
            app.update_status("Starting MARL training...")

        # Run training with proper error handling
        try:
            logger.info(f"Calling train_agents with {episodes} episodes, max_steps=100")

            # Store the result but don't use it directly to avoid NoneType errors
            training_result = app.marl_integration['marl'].train_agents(
                grid_state=grid_state,
                num_episodes=episodes,
                max_steps=100,
                save_interval=10,
                print_interval=2
            )

            logger.info("MARL training completed successfully")
            logger.debug(f"Training result: {training_result is not None}")

            # Signal completion
            if has_root:
                app.root.after(0, lambda: training_completed(app, True))
            else:
                logger.warning("Cannot signal completion: app has no root")

        except Exception as e:
            logger.error(f"Error in train_agents call: {e}")
            traceback.print_exc()

            if hasattr(app, 'update_status'):
                app.update_status(f"Error in MARL training: {str(e)}")

            if has_root:
                app.root.after(0, lambda: training_completed(app, False))

            raise  # Re-raise to be caught by outer try-except

    except Exception as e:
        logger.error(f"Error during training: {e}")
        traceback.print_exc()

        # Use root.after to safely update UI from a background thread
        if hasattr(app, 'update_status'):
            if has_root:
                app.root.after(0, lambda: app.update_status(f"Error during MARL training: {str(e)}"))
            else:
                logger.warning("Cannot update status: app has no root")

        if has_root:
            app.root.after(0, lambda: training_completed(app, False))
        else:
            logger.warning("Cannot signal completion: app has no root")

def update_training_progress(app):
    """Update the training progress bar and status."""
    logger.debug("Updating training progress")

    # Check if MARL integration is properly initialized
    if not hasattr(app, 'marl_integration') or app.marl_integration is None:
        logger.error("MARL integration not initialized in update_training_progress")
        return

    # Check if app.marl_integration is a dictionary (from this module)
    if not isinstance(app.marl_integration, dict):
        logger.error("App.marl_integration is not a dictionary, it might be from another module")
        return

    # Check if app has root for UI updates
    has_root = hasattr(app, 'root')
    if not has_root:
        logger.warning("App does not have 'root' attribute, UI updates will be skipped")
        return

    # Check if training is active
    if 'training_active' not in app.marl_integration or not app.marl_integration['training_active']:
        logger.debug("Training is not active, skipping progress update")
        return

    try:
        # Estimate progress based on saved models
        model_dir = "marl_models"
        if os.path.exists(model_dir):
            # Count model files
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
            if model_files:
                # Estimate progress
                progress = min(95, len(model_files) * 10)  # Rough estimate
                logger.debug(f"Found {len(model_files)} model files, progress: {progress}%")

                # Safely update progress
                if 'training_progress' in app.marl_integration:
                    app.marl_integration['training_progress'] = progress
                    logger.debug(f"Updated training_progress to {progress}%")

        # Safely update progress bar
        if ('progress_bar' in app.marl_integration and
            'training_progress' in app.marl_integration):
            try:
                app.marl_integration['progress_bar']['value'] = app.marl_integration['training_progress']
                logger.debug(f"Updated progress bar to {app.marl_integration['training_progress']}%")
            except Exception as e:
                logger.error(f"Error updating progress bar: {e}")

        # Safely update status label
        if ('status_label' in app.marl_integration and
            'training_progress' in app.marl_integration):
            try:
                if app.marl_integration['training_progress'] < 100:
                    app.marl_integration['status_label'].config(
                        text=f"{app.marl_integration['training_progress']}% complete"
                    )
                    logger.debug(f"Updated status label to {app.marl_integration['training_progress']}% complete")
                else:
                    app.marl_integration['status_label'].config(text="Finalizing...")
                    logger.debug("Updated status label to Finalizing...")
            except Exception as e:
                logger.error(f"Error updating status label: {e}")

        # Schedule next update if training is still active
        if 'training_active' in app.marl_integration and app.marl_integration['training_active']:
            logger.debug("Scheduling next progress update")
            try:
                app.root.after(500, lambda: update_training_progress(app))
            except Exception as e:
                logger.error(f"Error scheduling next update: {e}")
        else:
            logger.debug("Training is no longer active, not scheduling next update")

    except Exception as e:
        logger.error(f"Error updating training progress: {e}")
        traceback.print_exc()

        # Still try to schedule the next update
        try:
            if has_root:
                app.root.after(500, lambda: update_training_progress(app))
        except Exception as e2:
            logger.error(f"Error scheduling next update after exception: {e2}")

def training_completed(app, success):
    """Called when training is completed."""
    logger.info(f"Training completed with success={success}")

    # Check if MARL integration is properly initialized
    if not hasattr(app, 'marl_integration') or app.marl_integration is None:
        logger.error("MARL integration not initialized in training_completed")
        if hasattr(app, 'update_status'):
            app.update_status("Error: MARL integration not initialized")
        return

    # Check if app.marl_integration is a dictionary (from this module)
    if not isinstance(app.marl_integration, dict):
        logger.error("App.marl_integration is not a dictionary, it might be from another module")
        if hasattr(app, 'update_status'):
            app.update_status("Error: MARL integration is not from Simple_MARL_Integration")
        return

    # Check if app has root for UI updates
    has_root = hasattr(app, 'root')
    if not has_root:
        logger.warning("App does not have 'root' attribute, UI updates will be skipped")

    # Safely update training active flag
    if 'training_active' in app.marl_integration:
        logger.debug("Setting training_active to False")
        app.marl_integration['training_active'] = False
    else:
        logger.warning("training_active key not found in marl_integration")

    # Safely enable buttons
    try:
        if 'train_btn' in app.marl_integration:
            logger.debug("Enabling train button")
            app.marl_integration['train_btn'].config(state=tk.NORMAL)
        else:
            logger.warning("train_btn key not found in marl_integration")

        # Enable the main do_shape button if it exists
        if hasattr(app, 'do_shape_btn'):
            logger.debug("Enabling do_shape button")
            app.do_shape_btn.config(state=tk.NORMAL)
        else:
            logger.warning("do_shape_btn attribute not found in app")

        if success:
            logger.info("Training was successful, updating UI accordingly")

            # Enable MARL button if it exists
            if 'use_marl_btn' in app.marl_integration:
                logger.debug("Enabling use_marl button")
                app.marl_integration['use_marl_btn'].config(state=tk.NORMAL)
            else:
                logger.warning("use_marl_btn key not found in marl_integration")

            # Update progress and status if they exist
            if 'progress_bar' in app.marl_integration:
                logger.debug("Setting progress bar to 100%")
                app.marl_integration['progress_bar']['value'] = 100
            else:
                logger.warning("progress_bar key not found in marl_integration")

            if 'status_label' in app.marl_integration:
                logger.debug("Setting status label to 'Training complete'")
                app.marl_integration['status_label'].config(text="Training complete")
            else:
                logger.warning("status_label key not found in marl_integration")

            # Update parent status
            if hasattr(app, 'update_status'):
                logger.debug("Updating app status to success message")
                app.update_status("MARL training completed successfully!")
            else:
                logger.warning("update_status method not found in app")
        else:
            logger.info("Training failed, updating UI accordingly")

            # Update status if elements exist
            if 'progress_bar' in app.marl_integration:
                logger.debug("Setting progress bar to 0%")
                app.marl_integration['progress_bar']['value'] = 0
            else:
                logger.warning("progress_bar key not found in marl_integration")

            if 'status_label' in app.marl_integration:
                logger.debug("Setting status label to 'Training failed'")
                app.marl_integration['status_label'].config(text="Training failed")
            else:
                logger.warning("status_label key not found in marl_integration")

            # Update parent status
            if hasattr(app, 'update_status'):
                logger.debug("Updating app status to failure message")
                app.update_status("MARL training failed. See console for details.")
            else:
                logger.warning("update_status method not found in app")
    except Exception as e:
        logger.error(f"Error updating UI after training: {e}")
        traceback.print_exc()

        if hasattr(app, 'update_status'):
            app.update_status(f"Error updating UI after training: {str(e)}")

    logger.info("Training completion handling finished")

def do_shape_with_marl(app):
    """Use the trained MARL system to do the shape."""
    logger.info("do_shape_with_marl called")

    # Check if MARL integration is properly initialized
    if not hasattr(app, 'marl_integration') or app.marl_integration is None:
        logger.error("MARL integration not initialized in do_shape_with_marl")
        if hasattr(app, 'update_status'):
            app.update_status("Error: MARL integration not initialized")
        return

    # Check if app.marl_integration is a dictionary (from this module)
    if not isinstance(app.marl_integration, dict):
        logger.error("App.marl_integration is not a dictionary, it might be from another module")
        if hasattr(app, 'update_status'):
            app.update_status("Error: MARL integration is not from Simple_MARL_Integration")
        return

    # Check if app has root for UI updates
    has_root = hasattr(app, 'root')
    if not has_root:
        logger.warning("App does not have 'root' attribute, UI updates will be skipped")
        return

    # Check if MARL model is initialized
    if 'marl' not in app.marl_integration or app.marl_integration['marl'] is None:
        logger.error("MARL model not initialized")
        app.update_status("Please train MARL first!")
        return

    # Check if movement is already in progress
    if not hasattr(app, 'movement_started'):
        logger.error("App does not have 'movement_started' attribute")
        app.update_status("Error: App does not have movement_started attribute")
        return

    if app.movement_started:
        logger.warning("Movement already in progress")
        app.update_status("Movement already in progress!")
        return

    # Set movement started flag
    app.movement_started = True
    logger.debug(f"Movement started flag set to {app.movement_started}")

    # Disable buttons safely
    try:
        if 'train_btn' in app.marl_integration:
            logger.debug("Disabling train button")
            app.marl_integration['train_btn'].config(state=tk.DISABLED)

        if 'use_marl_btn' in app.marl_integration:
            logger.debug("Disabling use_marl button")
            app.marl_integration['use_marl_btn'].config(state=tk.DISABLED)

        if hasattr(app, 'do_shape_btn'):
            logger.debug("Disabling do_shape button")
            app.do_shape_btn.config(state=tk.DISABLED)
    except Exception as e:
        logger.error(f"Error disabling buttons: {e}")
        # Continue anyway

    # Check if app has the required attributes for grid state
    if not hasattr(app, 'cells'):
        logger.error("App does not have 'cells' attribute")
        app.update_status("Error: App does not have cells attribute")
        app.movement_started = False
        _re_enable_buttons(app)
        return

    # Convert grid to MARL format
    logger.debug("Converting grid to MARL format")
    grid_state = {}
    try:
        for cell in app.cells:
            grid_state[cell] = {
                "obstacle": app.cells[cell]["obstacle"],
                "active": app.cells[cell]["active"]
            }
        logger.debug(f"Grid state created with {len(grid_state)} cells")
    except Exception as e:
        logger.error(f"Error creating grid state: {e}")
        app.update_status(f"Error creating grid state: {str(e)}")
        app.movement_started = False
        _re_enable_buttons(app)
        return

    # Check if app has active cells and target shape
    if not hasattr(app, 'active_cells'):
        logger.error("App does not have 'active_cells' attribute")
        app.update_status("Error: App does not have active_cells attribute")
        app.movement_started = False
        _re_enable_buttons(app)
        return

    if not hasattr(app, 'target_shape'):
        logger.error("App does not have 'target_shape' attribute")
        app.update_status("Error: App does not have target_shape attribute")
        app.movement_started = False
        _re_enable_buttons(app)
        return

    # Get agent positions and targets
    agent_positions = app.active_cells.copy()
    target_positions = app.target_shape[:len(agent_positions)]

    logger.info(f"Agent positions: {agent_positions}")
    logger.info(f"Target positions: {target_positions}")

    if not agent_positions:
        logger.warning("No active cells found")
        app.update_status("No active cells found!")
        app.movement_started = False
        _re_enable_buttons(app)
        return

    if not target_positions:
        logger.warning("No target shape defined")
        app.update_status("No target shape defined!")
        app.movement_started = False
        _re_enable_buttons(app)
        return

    # Identify the shape border cells
    try:
        # Get the shape border cells
        shape_border = _identify_shape_border(app.target_shape)
        logger.info(f"Identified {len(shape_border)} border cells in the target shape")

        # Store the shape border for later use
        app.marl_integration['shape_border'] = shape_border
    except Exception as e:
        logger.error(f"Error identifying shape border: {e}")
        traceback.print_exc()
        # Continue without shape border information
        app.marl_integration['shape_border'] = []

    # Sort agent positions and target positions by difficulty
    # This ensures cells move in a queue with the first cell going to the hardest target
    try:
        agent_positions, target_positions = _sort_by_difficulty(agent_positions, target_positions, grid_state)
        logger.info("Sorted agent and target positions by difficulty")
    except Exception as e:
        logger.error(f"Error sorting positions by difficulty: {e}")
        traceback.print_exc()
        # Continue with unsorted positions

    # Plan paths using MARL
    app.update_status("Planning paths with MARL...")
    logger.info("Planning paths with MARL...")

    try:
        # Call plan_paths with proper error handling
        logger.debug("Calling plan_paths")
        result = app.marl_integration['marl'].plan_paths(
            grid_state,
            agent_positions,
            target_positions,
            max_steps=100,
            training=False
        )

        logger.debug(f"MARL plan_paths result: {result is not None}")

        if result is None:
            logger.error("plan_paths returned None")
            app.update_status("MARL path planning failed!")
            app.movement_started = False
            _re_enable_buttons(app)
            return

        logger.debug(f"Paths found: {len(result['paths'])}")
        logger.debug(f"Success: {result['success']}")
        logger.debug(f"Rewards: {result['rewards']}")

        # Use the paths to move agents
        if result['paths']:
            app.update_status("Starting movement with MARL paths")
            logger.info("Starting movement with MARL paths")

            # Check if app has moving_cells attribute
            if not hasattr(app, 'moving_cells'):
                logger.error("App does not have 'moving_cells' attribute")
                app.update_status("Error: App does not have moving_cells attribute")
                app.movement_started = False
                _re_enable_buttons(app)
                return

            # Initialize moving cells
            app.moving_cells = {}

            # Store the queue order for reference
            app.marl_integration['queue_order'] = []

            # Start all movements
            paths_added = 0
            for i, (agent_pos, target_pos, path) in enumerate(zip(agent_positions, target_positions, result['paths'])):
                if not path:
                    logger.warning(f"Empty path for agent {i}")
                    continue

                # Generate a unique ID for this movement
                move_id = f"marl_{i}"

                # Add to queue order
                app.marl_integration['queue_order'].append(move_id)

                # Store the path and current index
                # Format: (path, current_index, target_pos, start_pos, in_shape_border)
                app.moving_cells[move_id] = (path, 1, target_pos, agent_pos, False)

                # Remove cell from active cells
                if agent_pos in app.active_cells:
                    app.active_cells.remove(agent_pos)

                paths_added += 1
                logger.debug(f"Added path for agent {i}: {agent_pos} -> {target_pos}, path length: {len(path)}")

            logger.info(f"Total paths added: {paths_added}")
            logger.info(f"Moving cells: {len(app.moving_cells)}")

            # Override the update_moving_cells method to handle queue movement and border detection
            if hasattr(app, 'update_moving_cells'):
                # Store the original method
                if not hasattr(app, '_original_update_moving_cells'):
                    app._original_update_moving_cells = app.update_moving_cells

                # Override with our custom method
                app.update_moving_cells = lambda: _custom_update_moving_cells(app)
                logger.info("Overrode update_moving_cells method to handle queue movement")

            if paths_added > 0:
                # Check if app has movement_speed attribute
                if not hasattr(app, 'movement_speed'):
                    logger.error("App does not have 'movement_speed' attribute")
                    app.update_status("Error: App does not have movement_speed attribute")
                    app.movement_started = False
                    _re_enable_buttons(app)
                    return

                # Start the movement timer
                logger.debug(f"Starting movement timer with speed: {app.movement_speed}")
                app.movement_timer = app.root.after(app.movement_speed, app.update_moving_cells)
                logger.debug(f"Movement timer ID: {app.movement_timer}")
            else:
                logger.warning("No valid paths found")
                app.update_status("No valid paths found!")
                app.movement_started = False
                _re_enable_buttons(app)
        else:
            logger.warning("MARL path planning failed (no paths)")
            app.update_status("MARL path planning failed!")
            app.movement_started = False
            _re_enable_buttons(app)
    except Exception as e:
        logger.error(f"Exception in MARL path planning: {e}")
        traceback.print_exc()

        if hasattr(app, 'update_status'):
            app.update_status(f"Error in MARL path planning: {str(e)}")

        app.movement_started = False
        _re_enable_buttons(app)

# Helper function to identify the border cells of a shape
def _identify_shape_border(shape_cells):
    """
    Identify the border cells of a shape.

    Args:
        shape_cells: List of (row, col) tuples representing the shape

    Returns:
        List of (row, col) tuples representing the border cells
    """
    if not shape_cells:
        return []

    # Convert shape cells to a set for faster lookups
    shape_set = set(shape_cells)
    border_cells = []

    # Directions for checking neighbors (up, right, down, left)
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    # Check each cell in the shape
    for cell in shape_cells:
        row, col = cell

        # Check if any of the cell's neighbors is not in the shape
        for dr, dc in directions:
            neighbor = (row + dr, col + dc)
            if neighbor not in shape_set:
                # This cell is on the border
                border_cells.append(cell)
                break

    return border_cells

# Helper function to sort agent and target positions by difficulty
def _sort_by_difficulty(agent_positions, target_positions, grid_state):
    """
    Sort agent and target positions by difficulty.
    The most difficult target gets the first agent, and so on.

    Args:
        agent_positions: List of (row, col) tuples representing agent positions
        target_positions: List of (row, col) tuples representing target positions
        grid_state: Dictionary of cell states

    Returns:
        Tuple of (sorted_agent_positions, sorted_target_positions)
    """
    if not agent_positions or not target_positions:
        return agent_positions, target_positions

    # Calculate difficulty for each target position
    target_difficulties = []

    for target in target_positions:
        # Factors that contribute to difficulty:
        # 1. Distance from the center of the grid
        # 2. Number of obstacles nearby
        # 3. Number of other targets nearby

        # Calculate grid center
        grid_size = max(max(row for row, _ in grid_state.keys()),
                        max(col for _, col in grid_state.keys())) + 1
        center_row, center_col = grid_size // 2, grid_size // 2

        # Calculate distance from center
        distance_from_center = abs(target[0] - center_row) + abs(target[1] - center_col)

        # Count obstacles nearby
        obstacles_nearby = 0
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                neighbor = (target[0] + dr, target[1] + dc)
                if neighbor in grid_state and grid_state[neighbor].get("obstacle", False):
                    obstacles_nearby += 1

        # Count other targets nearby
        targets_nearby = 0
        for other_target in target_positions:
            if other_target != target:
                distance = abs(target[0] - other_target[0]) + abs(target[1] - other_target[1])
                if distance <= 2:
                    targets_nearby += 1

        # Calculate total difficulty score
        difficulty = distance_from_center + obstacles_nearby * 2 + targets_nearby
        target_difficulties.append((target, difficulty))

    # Sort targets by difficulty (highest first)
    target_difficulties.sort(key=lambda x: x[1], reverse=True)
    sorted_targets = [t[0] for t in target_difficulties]

    # Sort agents by their distance to the first target
    agent_distances = []
    first_target = sorted_targets[0]

    for agent in agent_positions:
        distance = abs(agent[0] - first_target[0]) + abs(agent[1] - first_target[1])
        agent_distances.append((agent, distance))

    # Sort agents by distance (closest first)
    agent_distances.sort(key=lambda x: x[1])
    sorted_agents = [a[0] for a in agent_distances]

    return sorted_agents, sorted_targets

# Helper function for custom update of moving cells with queue movement and border detection
def _custom_update_moving_cells(app):
    """
    Custom implementation of update_moving_cells that handles queue movement
    and allows cells to break from the queue when they reach the shape border.
    """
    logger.debug("Custom update_moving_cells called")

    # Check if we have shape border information
    has_shape_border = hasattr(app, 'marl_integration') and 'shape_border' in app.marl_integration
    shape_border = app.marl_integration.get('shape_border', []) if has_shape_border else []
    shape_border_set = set(shape_border)

    # Check if we have queue order information
    has_queue_order = hasattr(app, 'marl_integration') and 'queue_order' in app.marl_integration
    queue_order = app.marl_integration.get('queue_order', []) if has_queue_order else []

    # First pass - collect all planned moves and detect collisions
    completed_moves = []
    next_positions = {}
    collision_cells = set()

    # Process cells in queue order
    for move_id in queue_order:
        if move_id not in app.moving_cells:
            continue

        path, index, target, start_cell, in_shape_border = app.moving_cells[move_id]

        if index >= len(path):
            # Movement complete
            completed_moves.append(move_id)
            continue

        # Get current and next positions
        current_pos = path[index - 1] if index > 0 else start_cell
        next_pos = path[index]

        # Check if the cell has reached the shape border
        if not in_shape_border and next_pos in shape_border_set:
            logger.info(f"Cell {move_id} reached shape border at {next_pos}")
            # Mark as in shape border
            app.moving_cells[move_id] = (path, index, target, start_cell, True)
            in_shape_border = True

        # If the cell is in the shape border, it can move freely to its target
        # Otherwise, it follows the queue rules
        if in_shape_border:
            # Check for collisions with other cells
            if next_pos in next_positions:
                collision_cells.add(next_pos)
            else:
                next_positions[next_pos] = move_id
        else:
            # Follow queue rules - only move if the cell ahead has moved
            can_move = True

            # Find the cell's position in the queue
            queue_pos = queue_order.index(move_id)

            # Check if there's a cell ahead in the queue
            if queue_pos > 0:
                cell_ahead = queue_order[queue_pos - 1]

                # Check if the cell ahead exists and has moved
                if cell_ahead in app.moving_cells:
                    ahead_path, ahead_index, ahead_target, ahead_start, ahead_in_border = app.moving_cells[cell_ahead]

                    # If the cell ahead hasn't moved or is too close, don't move
                    if not ahead_in_border:  # Only apply queue rules if the cell ahead is not in the border
                        ahead_current = ahead_path[ahead_index - 1] if ahead_index > 0 else ahead_start
                        distance = abs(current_pos[0] - ahead_current[0]) + abs(current_pos[1] - ahead_current[1])

                        if distance < 2:  # Maintain a minimum distance of 2 cells
                            can_move = False

            if can_move:
                # Check for collisions with other cells
                if next_pos in next_positions:
                    collision_cells.add(next_pos)
                else:
                    next_positions[next_pos] = move_id

    # Second pass - execute non-colliding moves
    for move_id in list(app.moving_cells.keys()):
        if move_id in completed_moves:
            # Remove completed movements
            del app.moving_cells[move_id]
            continue

        path, index, target, start_cell, in_shape_border = app.moving_cells[move_id]

        if index >= len(path):
            # Movement complete
            del app.moving_cells[move_id]
            continue

        next_pos = path[index]
        current_pos = path[index - 1] if index > 0 else start_cell

        # Skip moves to collision cells
        if next_pos in collision_cells:
            continue

        # Check if this move is allowed
        if next_pos in next_positions and next_positions[next_pos] == move_id:
            # Move the cell
            try:
                # Deactivate current cell
                if hasattr(app, 'deactivate_cell'):
                    app.deactivate_cell(current_pos)

                # Activate next cell
                if hasattr(app, 'activate_cell'):
                    app.activate_cell(next_pos)

                # Update the moving cell's index
                app.moving_cells[move_id] = (path, index + 1, target, start_cell, in_shape_border)

                logger.debug(f"Moved cell {move_id} from {current_pos} to {next_pos}")

                # Check if we've reached the target
                if next_pos == target:
                    logger.info(f"Cell {move_id} reached target {target}")
                    # Mark as completed
                    completed_moves.append(move_id)
            except Exception as e:
                logger.error(f"Error moving cell {move_id}: {e}")
                traceback.print_exc()

    # Remove completed movements
    for move_id in completed_moves:
        if move_id in app.moving_cells:
            del app.moving_cells[move_id]

    # Check if all movements are complete
    if not app.moving_cells:
        logger.info("All movements complete")

        # Reset movement started flag
        app.movement_started = False

        # Re-enable buttons
        _re_enable_buttons(app)

        # Restore original update_moving_cells method if it exists
        if hasattr(app, '_original_update_moving_cells'):
            app.update_moving_cells = app._original_update_moving_cells
            delattr(app, '_original_update_moving_cells')
            logger.info("Restored original update_moving_cells method")

        # Update status
        if hasattr(app, 'update_status'):
            app.update_status("MARL shape formation complete!")

        return

    # Schedule the next update
    if hasattr(app, 'root') and hasattr(app, 'movement_speed'):
        app.movement_timer = app.root.after(app.movement_speed, app.update_moving_cells)

# Helper function to re-enable buttons
def _re_enable_buttons(app):
    """Re-enable buttons after an error or completion."""
    logger.debug("Re-enabling buttons")

    # Check if MARL integration is properly initialized
    if not hasattr(app, 'marl_integration') or app.marl_integration is None:
        logger.error("MARL integration not initialized in _re_enable_buttons")
        return

    # Check if app.marl_integration is a dictionary (from this module)
    if not isinstance(app.marl_integration, dict):
        logger.error("App.marl_integration is not a dictionary, it might be from another module")
        return

    try:
        if 'train_btn' in app.marl_integration:
            logger.debug("Enabling train button")
            app.marl_integration['train_btn'].config(state=tk.NORMAL)

        if 'use_marl_btn' in app.marl_integration:
            logger.debug("Enabling use_marl button")
            app.marl_integration['use_marl_btn'].config(state=tk.NORMAL)

        if hasattr(app, 'do_shape_btn'):
            logger.debug("Enabling do_shape button")
            app.do_shape_btn.config(state=tk.NORMAL)
    except Exception as e:
        logger.error(f"Error re-enabling buttons: {e}")
        # Nothing more we can do here
