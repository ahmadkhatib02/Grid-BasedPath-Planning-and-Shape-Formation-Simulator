# Model Compatibility in MARL System

This document explains the model compatibility system implemented in the MARL shape formation project.

## Overview

The MARL system has been significantly enhanced with a new neural network architecture and state representation optimized for path blocking scenarios. These changes make previously trained models incompatible with the new system.

## Compatibility Mechanism

The following mechanisms have been implemented to handle model compatibility:

1. **Version Tracking**: Models are now saved with version information in a JSON file
   - Current version: 2 (enhanced state representation with path blocking)
   - Previous version: 1 (basic state representation)

2. **Compatibility Checking**: When loading models, the system checks:
   - Model version
   - State representation size
   - Grid size
   - Number of agents

3. **Automatic Backup**: When incompatible models are detected:
   - Old models are automatically backed up to a timestamped directory
   - New models are created with the current architecture

4. **Forced Reset**: When starting training, old models are backed up and removed
   - This ensures a clean start with the new architecture
   - Prevents mixing of incompatible model architectures

## What This Means for Users

When you run the system after updating to the new version:

1. Your old models will be automatically backed up
2. New models will be created with the enhanced architecture
3. You'll need to retrain the agents from scratch
4. The training will be more effective with the new architecture

## Technical Details

The key changes that caused the incompatibility:

1. **State Representation**: 
   - Old: Basic grid + agent/target positions (grid_size² + 8 values)
   - New: Enhanced grid + path information + detailed agent info (grid_size² + 19 values)

2. **Neural Network Architecture**:
   - Old: Simple DQN with basic layers
   - New: Dueling DQN with specialized processing for different state components

3. **Loss Function**:
   - Old: Mean Squared Error (MSE)
   - New: Huber Loss for better stability

## Backup Locations

Old models are backed up to:
- Regular training: `marl_models_backup_[timestamp]`
- Path blocking training: `marl_models_backup_pathblocking_[timestamp]`

## Future Compatibility

Future updates will maintain backward compatibility where possible, but significant architectural changes may require similar migration processes. The version tracking system will help manage these transitions.
