"""
Configuration file for PULearningPC parametric experiments.
Modify the values below to customize your experiments.
"""

# ============================================================================
# EXPERIMENT SETTINGS
# ============================================================================
EXPERIMENT_CONFIG = {
    'n_runs': 3,                    # Number of runs per parameter combination
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
    'pubmed': {
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
    }
}

# ============================================================================
# BASELINE MODELS CONFIGURATION
# ============================================================================
BASELINE_CONFIG = {
    'datasets': ['cora', 'citeseer', 'mnist', 'twitch', 'pubmed'],
    'percent_positive': [0.01, 0.25],
    'num_neg': {
        'cora': 200,
        'citeseer': 200,
        'mnist': 300,
        'twitch': 200,
        'pubmed': 200,
    },
    'models': {
        'RCSVM': {'alpha': 0.7, 'beta': 0.3},
        'CCRNE': {'ratio': 0.3},
        'PU_LP': {'alpha': 0.1, 'm': 3, 'l': 1},
        'MCLS': {'k': 7, 'ratio': 0.1},
        'LP_PUL': {},
    }
}

# ============================================================================
# PARAMETER RANGES TO TEST
# ============================================================================
PARAMETER_RANGES = {
    # Particle Competition Model parameters
    'num_particles': [2, 5, 10, 20, 50, 100],

    # Reliable Negative Selection parameters
    'cluster_strategy': ['majority', 'percentage'],
    'positive_cluster_threshold': [0.1, 0.2, 0.3, 0.4, 0.5],
    
    # Movement and initialization parameters
    'movement_strategy': ['uniform', 'degree_weighted'],
    'initialization_strategy': ['random', 'degree_weighted'],
    'avg_node_pot_threshold': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}

# ============================================================================
# REDUCED PARAMETER SETS FOR QUICK TESTING
# ============================================================================
QUICK_TEST_PARAMS = {
    'num_particles': [100, 200, 387],
    'cluster_strategy': ['majority', 'percentage'],
    'positive_cluster_threshold': [0.1, 0.3, 0.5],
    'movement_strategy': ['uniform', 'degree_weighted'],
    'initialization_strategy': ['random', 'degree_weighted'],
    'avg_node_pot_threshold': [0.7, 0.8, 0.9]
}

# ============================================================================
# PER-DATASET DEFAULT PARAMS FOR PULearningPC
# (rns_params for reliable negative selection + num_neg)
# ============================================================================
PULPC_DEFAULT_PARAMS = {
    'cora': {
        'rns_params': {
            'cluster_strategy': 'percentage',
            'positive_cluster_threshold': 0.1
        },
        'num_neg': 200
    },
    'citeseer': {
        'rns_params': {
            'cluster_strategy': 'percentage',
            'positive_cluster_threshold': 0.1
        },
        'num_neg': 200
    },
    'twitch': {
        'rns_params': {
            'cluster_strategy': 'percentage',
            'positive_cluster_threshold': 0.1
        },
        'num_neg': 200
    },
    'pubmed': {
        'rns_params': {
            'cluster_strategy': 'percentage',
            'positive_cluster_threshold': 0.1
        },
        'num_neg': 200
    },
    'mnist': {
        'rns_params': {
            'cluster_strategy': 'percentage',
            'positive_cluster_threshold': 0.1
        },
        'num_neg': 300
    }
}

# ============================================================================
# DATASET GENERATION PARAMETERS
# (used when generating datasets from scratch in experiment pipelines)
# ============================================================================
DATASET_GEN_PARAMS = {
    'cora': {
        'k': 3,
        'positive_class_label': 3,
        'use_original_edges': True,
        'mst': False
    },
    'citeseer': {
        'k': 3,
        'positive_class_label': 2,
        'use_original_edges': True,
        'mst': False
    },
    'pubmed': {
        'positive_class_label': 2,
        'use_original_edges': True,
        'mst': False
    },
    'twitch': {
        'mst': False
    },
    'mnist': {
        'k': 3,
        'mst': False,
        'n_samples': 3000
    }
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================
EVALUATION_METRICS = [
    'step1_f1_score',
    'step1_precision',
    'step1_recall',
    'step1_num_reliable_negatives',
    'step2_f1',
    'step2_precision',
    'step2_recall',
    'step2_accuracy'
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