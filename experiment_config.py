"""
Configuration file for PULearningPC parametric experiments.
Modify the values below to customize your experiments.
"""

# ============================================================================
# EXPERIMENT SETTINGS
# ============================================================================
EXPERIMENT_CONFIG = {
    'n_runs': 5,                    # Number of runs per parameter combination
    'output_dir': "experiment_results",
    'random_seed': 42,
    'save_intermediate': True,      # Save results every 10 experiments
    'num_reliable_negatives': 200   # Number of reliable negatives to select
}

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
DATASET_CONFIG = {
    'cora': {
        'k': 3,
        'positive_class_label': 3,
        'percent_positive': 0.1,
        'use_original_edges': True,
        'mst': False
    },
    'citeseer': {
        'positive_class_label': 2,
        'percent_positive': 0.1,
        'use_original_edges': True,
        'mst': False
    },
    'twitch': {
        'percent_positive': 0.1,
        'mst': False
    },
    'mnist': {
        'k': 3,
        'percent_positive': 0.1
    },
    'ionosphere': {
        'percent_positive': 0.1,
        'k': 3
    }
}

# ============================================================================
# PARAMETER RANGES TO TEST
# ============================================================================
PARAMETER_RANGES = {
    # Particle Competition Model parameters
    'num_particles': [50, 100, 200, 387, 500],
    'p_det': [0.6],  # Fixed at 0.6 as requested
    'delta_v': [0.1, 0.2, 0.3, 0.4, 0.5],
    'delta_p': [0.3, 0.5, 0.7, 0.8, 0.9],
    
    # Reliable Negative Selection parameters
    'cluster_strategy': ['majority', 'percentage'],
    'positive_cluster_threshold': [0.1, 0.3, 0.5, 0.7, 0.9],
    
    # Movement and initialization parameters
    'movement_strategy': ['uniform', 'degree_weighted'],
    'initialization_strategy': ['random', 'degree_weighted'],
    'avg_node_pot_threshold': [0.7, 0.8, 0.9]
}

# ============================================================================
# REDUCED PARAMETER SETS FOR QUICK TESTING
# ============================================================================
QUICK_TEST_PARAMS = {
    'num_particles': [100, 200, 387],
    'p_det': [0.6],
    'delta_v': [0.2, 0.3, 0.4],
    'delta_p': [0.5, 0.7, 0.8],
    'cluster_strategy': ['majority', 'percentage'],
    'positive_cluster_threshold': [0.1, 0.3, 0.5],
    'movement_strategy': ['uniform', 'degree_weighted'],
    'initialization_strategy': ['random', 'degree_weighted'],
    'avg_node_pot_threshold': [0.7, 0.8, 0.9]
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================
EVALUATION_METRICS = [
    'f1_score',
    'precision', 
    'recall',
    'num_reliable_negatives'
]

# ============================================================================
# OUTPUT FORMATS
# ============================================================================
OUTPUT_FORMATS = {
    'csv': True,
    'excel': False,
    'json': False,
    'pickle': False
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOGGING_CONFIG = {
    'level': 'INFO',
    'save_logs': True,
    'log_file': 'experiment_log.txt'
}

# ============================================================================
# PARALLEL PROCESSING (Future enhancement)
# ============================================================================
PARALLEL_CONFIG = {
    'use_parallel': False,
    'n_jobs': 1,
    'backend': 'multiprocessing'
} 