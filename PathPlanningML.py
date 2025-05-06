import numpy as np
import os
import pickle

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    print("TensorFlow or scikit-learn not available. ML features will be disabled.")
    ML_AVAILABLE = False

class PathPlanningML:
    """
    A class for applying supervised learning to path planning with obstacles.
    This class handles:
    1. Data collection from successful path planning scenarios
    2. Feature extraction from grid states
    3. Model training
    4. Path prediction
    """
    
    def __init__(self, grid_size=10):
        """
        Initialize the PathPlanningML class.
        
        Args:
            grid_size: Size of the grid (default: 10x10)
        """
        self.grid_size = grid_size
        self.model = None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.training_data = {
            'features': [],
            'targets': []
        }
        self.model_file = 'path_planning_model.h5'
        self.scaler_file = 'path_planning_scaler.pkl'
        
        # Try to load existing model and scaler
        self.load_model()
    
    def extract_features(self, grid_state, start_pos, target_pos):
        """
        Extract features from the current grid state.
        
        Args:
            grid_state: Dictionary of cell states
            start_pos: Starting position (row, col)
            target_pos: Target position (row, col)
            
        Returns:
            numpy array of features
        """
        if not ML_AVAILABLE:
            return None
            
        # Initialize feature vector
        features = []
        
        # Add start and target positions
        features.extend([start_pos[0] / self.grid_size, start_pos[1] / self.grid_size])
        features.extend([target_pos[0] / self.grid_size, target_pos[1] / self.grid_size])
        
        # Add Manhattan distance between start and target
        manhattan_dist = abs(start_pos[0] - target_pos[0]) + abs(start_pos[1] - target_pos[1])
        features.append(manhattan_dist / (2 * self.grid_size))
        
        # Add Euclidean distance between start and target
        euclidean_dist = np.sqrt((start_pos[0] - target_pos[0])**2 + (start_pos[1] - target_pos[1])**2)
        features.append(euclidean_dist / (np.sqrt(2) * self.grid_size))
        
        # Create obstacle map (1 for obstacle, 0 for free)
        obstacle_map = np.zeros((self.grid_size, self.grid_size))
        for pos, cell in grid_state.items():
            if cell.get("obstacle", False):
                obstacle_map[pos[0], pos[1]] = 1
        
        # Count obstacles in each quadrant relative to start position
        quadrant_obstacles = [0, 0, 0, 0]
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if obstacle_map[r, c] == 1:
                    # Determine quadrant (0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right)
                    quadrant = 0
                    if c >= start_pos[1]:
                        quadrant += 1
                    if r >= start_pos[0]:
                        quadrant += 2
                    quadrant_obstacles[quadrant] += 1
        
        # Normalize and add quadrant obstacle counts
        total_cells = self.grid_size * self.grid_size
        features.extend([count / total_cells for count in quadrant_obstacles])
        
        # Count obstacles along direct path from start to target
        direct_path_obstacles = 0
        steps = max(abs(start_pos[0] - target_pos[0]), abs(start_pos[1] - target_pos[1]))
        if steps > 0:
            for i in range(1, steps):
                r = int(start_pos[0] + (target_pos[0] - start_pos[0]) * i / steps)
                c = int(start_pos[1] + (target_pos[1] - start_pos[1]) * i / steps)
                if 0 <= r < self.grid_size and 0 <= c < self.grid_size and obstacle_map[r, c] == 1:
                    direct_path_obstacles += 1
            features.append(direct_path_obstacles / steps)
        else:
            features.append(0)
        
        # Add feature for whether there's a clear line of sight
        features.append(1.0 if direct_path_obstacles == 0 else 0.0)
        
        # Add density of obstacles in the region between start and target
        min_r, max_r = min(start_pos[0], target_pos[0]), max(start_pos[0], target_pos[0])
        min_c, max_c = min(start_pos[1], target_pos[1]), max(start_pos[1], target_pos[1])
        
        # Add padding to the region
        padding = 2
        min_r = max(0, min_r - padding)
        max_r = min(self.grid_size - 1, max_r + padding)
        min_c = max(0, min_c - padding)
        max_c = min(self.grid_size - 1, max_c + padding)
        
        region_size = (max_r - min_r + 1) * (max_c - min_c + 1)
        if region_size > 0:
            region_obstacles = np.sum(obstacle_map[min_r:max_r+1, min_c:max_c+1])
            features.append(region_obstacles / region_size)
        else:
            features.append(0)
        
        return np.array(features)
    
    def collect_training_data(self, grid_state, path, target_pos):
        """
        Collect training data from a successful path.
        
        Args:
            grid_state: Dictionary of cell states
            path: List of positions in the path
            target_pos: Target position
        """
        if not ML_AVAILABLE or not path:
            return
        
        # For each step in the path, extract features and the next direction
        for i in range(len(path) - 1):
            current_pos = path[i]
            next_pos = path[i + 1]
            
            # Extract features for this state
            features = self.extract_features(grid_state, current_pos, target_pos)
            
            # Determine the direction to move (0: up, 1: right, 2: down, 3: left)
            dr = next_pos[0] - current_pos[0]
            dc = next_pos[1] - current_pos[1]
            
            if dr == -1 and dc == 0:
                direction = 0  # Up
            elif dr == 0 and dc == 1:
                direction = 1  # Right
            elif dr == 1 and dc == 0:
                direction = 2  # Down
            elif dr == 0 and dc == -1:
                direction = 3  # Left
            else:
                # Diagonal moves or invalid moves are skipped
                continue
            
            # Add to training data
            self.training_data['features'].append(features)
            self.training_data['targets'].append(direction)
    
    def train_model(self, epochs=50, batch_size=32):
        """
        Train the model using collected data.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        if not ML_AVAILABLE or len(self.training_data['features']) < 100:
            print("Not enough training data or ML libraries not available")
            return None
        
        # Convert to numpy arrays
        X = np.array(self.training_data['features'])
        y = np.array(self.training_data['targets'])
        
        # One-hot encode the target directions
        y_onehot = tf.keras.utils.to_categorical(y, num_classes=4)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42)
        
        # Create model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(4, activation='softmax')  # 4 possible directions
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        # Save the model
        self.model = model
        self.save_model()
        
        return history
    
    def predict_next_move(self, grid_state, current_pos, target_pos):
        """
        Predict the next best move using the trained model.
        
        Args:
            grid_state: Dictionary of cell states
            current_pos: Current position (row, col)
            target_pos: Target position (row, col)
            
        Returns:
            Predicted next position (row, col) or None if prediction fails
        """
        if not ML_AVAILABLE or self.model is None:
            return None
        
        # Extract features
        features = self.extract_features(grid_state, current_pos, target_pos)
        if features is None:
            return None
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict direction probabilities
        direction_probs = self.model.predict(features_scaled)[0]
        
        # Get possible moves (avoid obstacles and stay in bounds)
        possible_moves = []
        directions = [
            (-1, 0),  # Up
            (0, 1),   # Right
            (1, 0),   # Down
            (0, -1)   # Left
        ]
        
        for i, (dr, dc) in enumerate(directions):
            new_r, new_c = current_pos[0] + dr, current_pos[1] + dc
            
            # Check if the move is valid
            if (0 <= new_r < self.grid_size and 
                0 <= new_c < self.grid_size and 
                not grid_state.get((new_r, new_c), {}).get("obstacle", False) and
                not grid_state.get((new_r, new_c), {}).get("active", False)):
                possible_moves.append((i, (new_r, new_c), direction_probs[i]))
        
        if not possible_moves:
            return None
        
        # Sort by probability (highest first)
        possible_moves.sort(key=lambda x: x[2], reverse=True)
        
        # Return the highest probability valid move
        return possible_moves[0][1]
    
    def save_model(self):
        """Save the model and scaler to files."""
        if not ML_AVAILABLE or self.model is None:
            return
        
        # Save model
        self.model.save(self.model_file)
        
        # Save scaler
        with open(self.scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_model(self):
        """Load the model and scaler from files if they exist."""
        if not ML_AVAILABLE:
            return
        
        # Load model if file exists
        if os.path.exists(self.model_file):
            try:
                self.model = load_model(self.model_file)
                print("Loaded existing path planning model")
            except Exception as e:
                print(f"Error loading model: {e}")
        
        # Load scaler if file exists
        if os.path.exists(self.scaler_file):
            try:
                with open(self.scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("Loaded existing feature scaler")
            except Exception as e:
                print(f"Error loading scaler: {e}")
