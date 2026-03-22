"""
Configuration settings for FitMetrics project
Central place for all parameters and paths
"""

# ============================================
# FILE PATHS
# ============================================

# Data paths
TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
ORIGINAL_PATH = 'data/calories.csv'

# Model paths
RF_MODEL = 'models/rf_model.pkl'
GB_MODEL = 'models/gb_model.pkl'
RIDGE_MODEL = 'models/ridge_model.pkl'

# Output paths
SUBMISSION_PATH = 'submission.csv'

# ============================================
# MODEL PARAMETERS
# ============================================

# General settings
RANDOM_STATE = 42
N_FOLDS = 5
TARGET = 'Calories'

# Random Forest parameters
RF_PARAMS = {
    'n_estimators': 100,        # Number of trees
    'max_depth': 10,            # Maximum depth of each tree
    'random_state': 42,         # For reproducibility
    'n_jobs': -1                # Use all CPU cores
}

# Gradient Boosting parameters
GB_PARAMS = {
    'n_estimators': 100,        # Number of boosting stages
    'learning_rate': 0.1,       # Step size for each iteration
    'max_depth': 5,             # Maximum depth of each tree
    'random_state': 42          # For reproducibility
}

# Ridge Regression parameters
RIDGE_PARAMS = {
    'alpha': 1.0,               # Regularization strength
    'random_state': 42          # For reproducibility
}

# ============================================
# ENSEMBLE WEIGHTS
# ============================================

# Weights for weighted averaging ensemble
# Based on cross-validation performance
WEIGHTS = {
    'rf': 0.4,      # Random Forest: 40%
    'gb': 0.4,      # Gradient Boosting: 40%
    'ridge': 0.2    # Ridge Regression: 20%
}

# ============================================
# FEATURE INFORMATION
# ============================================

# All features in the dataset
FEATURES = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

# Numeric features
NUMERIC_FEATURES = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

# Categorical features
CATEGORICAL_FEATURES = ['Sex']

# ============================================
# DATA PREPROCESSING
# ============================================

# Gender encoding mapping
GENDER_MAPPING = {
    'male': 0,
    'female': 1
}

# ============================================
# DISPLAY SETTINGS
# ============================================

# Number of decimal places for display
DISPLAY_DECIMALS = 4

# Visualization settings
FIGURE_SIZE = (12, 6)
DPI = 300  # Resolution for saved figures