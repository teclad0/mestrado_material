# PULearningPC Parametric Experiment Framework

This framework allows you to run comprehensive parametric experiments with your PULearningPC algorithm across multiple datasets and parameter combinations.

## 🚀 Quick Start

### 1. Basic Usage
```python
from parametric_experiments import PULearningExperimentRunner

# Initialize runner
runner = PULearningExperimentRunner(
    n_runs=5,                    # Number of runs per parameter combination
    output_dir="experiment_results",
    random_seed=42
)

# Run experiments on Cora dataset
results = runner.run_experiments('cora')
```

### 2. Run the Simple Script
```bash
python run_experiments.py
```

### 3. Run Examples
```bash
python example_usage.py
```

## 📁 Files Overview

- **`parametric_experiments.py`** - Main experiment framework
- **`run_experiments.py`** - Simple script to run experiments
- **`experiment_config.py`** - Configuration file for parameters
- **`example_usage.py`** - Examples showing different usage patterns
- **`README_experiments.md`** - This file

## ⚙️ Parameters Tested

The framework automatically tests all combinations of these parameters:

| Parameter | Values | Description |
|-----------|--------|-------------|
| `num_particles` | [50, 100, 200, 387, 500] | Number of particles in competition |
| `p_det` | [0.6] | **Fixed at 0.6** as requested |
| `delta_v` | [0.1, 0.2, 0.3, 0.4, 0.5] | Velocity decay parameter |
| `delta_p` | [0.3, 0.5, 0.7, 0.8, 0.9] | Potential decay parameter |
| `cluster_strategy` | ['majority', 'percentage'] | How to label clusters |
| `positive_cluster_threshold` | [0.1, 0.3, 0.5, 0.7, 0.9] | Threshold for positive clusters |
| `movement_strategy` | ['uniform', 'degree_weighted'] | Particle movement strategy |
| `initialization_strategy` | ['random', 'degree_weighted'] | Particle initialization strategy |
| `avg_node_pot_threshold` | [0.7, 0.8, 0.9] | Stopping threshold |

## 🎯 Datasets Supported

- **Cora** - Citation network
- **CiteSeer** - Citation network  
- **Twitch** - Social network
- **MNIST** - Image data (converted to graph)
- **Ionosphere** - Feature data (converted to graph)

## 📊 Output Files

For each dataset, the framework generates:

1. **`{dataset}_final_results.csv`** - All experiment results
2. **`{dataset}_summary_results.csv`** - Aggregated results by parameter combination
3. **`{dataset}_intermediate_results.csv`** - Intermediate saves (every 10 experiments)

### Understanding the Fallback Rule

The **`fallback_rule_used`** column indicates whether the algorithm had to use the fallback rule during cluster labeling:

- **`False`**: The algorithm found positive clusters using the normal strategy (majority or percentage)
- **`True`**: No positive clusters were found, so the algorithm used the fallback rule to select the cluster with the highest positive ratio as "positive"

This information helps you understand:
- When your parameters are too strict (fallback rule used frequently)
- When your parameters are working well (fallback rule rarely used)
- The robustness of your clustering strategy

### Results Include:
- All parameter values
- F1 score, precision, recall
- Number of reliable negatives selected
- Coverage percentage
- Fallback rule usage (whether the algorithm had to use fallback rule)
- Graph statistics (nodes, edges)
- Run status (success/error)

## 🔧 Customization

### Modify Parameter Ranges
Edit `experiment_config.py`:
```python
PARAMETER_RANGES = {
    'num_particles': [100, 200, 387],  # Your custom values
    'delta_v': [0.2, 0.3, 0.4],       # Your custom values
    # ... other parameters
}
```

### Custom Dataset Parameters
```python
DATASET_CONFIG = {
    'cora': {
        'k': 5,                        # Custom k for k-NN
        'percent_positive': 0.2,       # Custom positive ratio
        'use_original_edges': True,
        'mst': False
    }
}
```

### Quick Testing
Use reduced parameter sets for faster testing:
```python
# In run_experiments.py
CUSTOM_PARAM_RANGES = {
    'num_particles': [100, 200],      # Only 2 values
    'delta_v': [0.2, 0.4],           # Only 2 values
    # ... other reduced ranges
}
```

## 📈 Analysis Examples

### Load and Analyze Results
```python
import pandas as pd

# Load results
results = pd.read_csv("experiment_results/cora_final_results.csv")

# Filter successful runs
successful = results[results['status'] == 'success']

# Find best parameters
best_run = successful.loc[successful['f1_score'].idxmax()]
print(f"Best F1: {best_run['f1_score']:.4f}")

# Parameter analysis
strategy_performance = successful.groupby('cluster_strategy')['f1_score'].agg(['mean', 'std'])
print(strategy_performance)
```

### Parameter Importance
```python
# Analyze impact of each parameter
for param in ['num_particles', 'delta_v', 'delta_p']:
    param_impact = successful.groupby(param)['f1_score'].mean()
    print(f"{param} impact:\n{param_impact}\n")
```

## ⚡ Performance Tips

1. **Start Small**: Use `QUICK_TEST_PARAMS` for initial testing
2. **Reduce Runs**: Start with `n_runs=2` or `n_runs=3`
3. **Test Single Dataset**: Focus on one dataset first
4. **Monitor Progress**: Results are saved every 10 experiments

## 🐛 Troubleshooting

### Common Issues:

1. **Memory Issues**: Reduce parameter ranges or number of runs
2. **Long Runtime**: Use smaller datasets or fewer parameter combinations
3. **Import Errors**: Ensure all dependencies are installed
4. **Graph Connectivity**: Check that `mst=False` is set for largest component

### Error Handling:
- Failed experiments are logged with error messages
- Results continue to be saved even if some experiments fail
- Check the `status` column for any errors

## 🔬 Advanced Usage

### Custom Evaluation Metrics
```python
def custom_evaluation(graph, reliable_negatives, ground_truth):
    # Your custom evaluation logic
    return {'custom_metric': value}

# Override in runner
runner.evaluate_reliable_negatives = custom_evaluation
```

### Parallel Processing (Future)
```python
# In experiment_config.py
PARALLEL_CONFIG = {
    'use_parallel': True,
    'n_jobs': 4,
    'backend': 'multiprocessing'
}
```

### Custom Stopping Criteria
```python
# Modify the ParticleCompetitionModel class
# or override the run_simulation method
```

## 📝 Example Workflow

1. **Setup**: Install dependencies, check dataset availability
2. **Test**: Run with `QUICK_TEST_PARAMS` to verify everything works
3. **Configure**: Adjust parameters in `experiment_config.py`
4. **Run**: Execute full experiments with `python run_experiments.py`
5. **Analyze**: Load results and analyze performance
6. **Iterate**: Refine parameters based on results

## 🤝 Contributing

To extend the framework:
- Add new evaluation metrics in `evaluate_reliable_negatives()`
- Add new datasets in `load_dataset()`
- Add new parameters in `get_parameter_grid()`
- Implement parallel processing for faster execution

## 📚 Dependencies

- `networkx` - Graph operations
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Evaluation metrics
- `tqdm` - Progress bars
- Your existing `model.py` and `generate_dataset.py`

## 🎯 Next Steps

1. Run the examples to understand the framework
2. Customize parameters for your specific needs
3. Run experiments on your datasets
4. Analyze results to find optimal parameters
5. Extend the framework as needed

Happy experimenting! 🚀 