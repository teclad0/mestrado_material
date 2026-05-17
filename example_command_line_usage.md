# Command Line Usage Examples

This document shows how to run experiments from the command line.

---

## Baseline Models (`run_baselines.py`)

Runs all baseline PU Learning models (RCSVM, CCRNE, PU_LP, MCLS, LP_PUL) with Phase 1 (reliable negative extraction) and Phase 2 (full classification).

### Basic Usage

```bash
# All models, all datasets, both percents (1% and 25% positive labeling)
python run_baselines.py

# Single dataset
python run_baselines.py --dataset cora

# Single model
python run_baselines.py --model RCSVM

# Single percent
python run_baselines.py --percent 0.25

# Combine filters
python run_baselines.py --dataset cora --model PU_LP --percent 0.01

# Custom output
python run_baselines.py --output my_results.csv
```

### Configuration

All baseline parameters are centralized in `experiment_config.py` under `BASELINE_CONFIG`:

```python
BASELINE_CONFIG = {
    'datasets': ['cora', 'citeseer', 'mnist', 'twitch', 'pubmed'],
    'percent_positive': [0.01, 0.25],
    'num_neg': {'cora': 200, 'citeseer': 200, 'mnist': 300, 'twitch': 200, 'pubmed': 200},
    'models': {
        'RCSVM': {'alpha': 0.7, 'beta': 0.3},
        'CCRNE': {'ratio': 0.3},
        'PU_LP': {'alpha': 0.1, 'm': 3, 'l': 1},
        'MCLS': {'k': 7, 'ratio': 0.1},
        'LP_PUL': {},
    }
}
```

### Output

Results are saved as CSV with columns:
- `model`, `dataset`, `percent_positive`
- Phase 1: `phase1_f1`, `phase1_precision`, `phase1_num_rn`, `phase1_time_s`
- Phase 2: `phase2_f1`, `phase2_precision`, `phase2_recall`, `phase2_accuracy`, `phase2_time_s`

### Available Models

| Model | Phase 1 (RN extraction) | Phase 2 (Classification) |
|-------|------------------------|--------------------------|
| RCSVM | Cosine-similarity representant vectors | Iterative SVM expansion |
| CCRNE | Clustering with radius threshold | Weighted Voting SVM |
| PU_LP | Katz similarity (sparse solve) | Label Propagation |
| MCLS | KMeans + cluster labeling | Iterative LS-SVM |
| LP_PUL | BFS distance from positives | Harmonic function propagation |

### Notes

- Datasets must exist in `datasets/` as pickle files (e.g., `cora_full_1pct.pkl`)
- Generate missing datasets with: `python dataset_system.py --dataset <name>`
- PU_LP uses sparse linear solve — works on PubMed (19717 nodes) without memory issues
- CCRNE/RCSVM Phase 2 uses iterative SVMs — slower on large datasets (~60-80s on Cora)

---

## PULearningPC Parametric Experiments (`run_experiments.py`)

This runs parametric grid search over the CP-APNR (PULearningPC) hyperparameters.

## 🚀 Basic Usage

### 1. Run experiments on Cora dataset with default parameters
```bash
python run_experiments.py --dataset cora
```

### 2. Run experiments with custom number of runs
```bash
python run_experiments.py --dataset cora --n-runs 5
```

### 3. Run experiments on multiple datasets
```bash
python run_experiments.py --dataset cora citeseer twitch
```

### 4. Quick test with reduced parameters (faster execution)
```bash
python run_experiments.py --dataset cora --quick-test
```

## ⚙️ Custom Parameter Ranges

### 1. Test specific number of particles
```bash
python run_experiments.py --dataset cora --num-particles 100 200 387
```

### 2. Test specific cluster strategies
```bash
python run_experiments.py --dataset cora --cluster-strategy majority percentage
```

### 3. Test specific thresholds
```bash
python run_experiments.py --dataset cora --positive-cluster-threshold 0.01 0.1 0.3
```

## 🎯 Advanced Usage

### 1. Custom output directory
```bash
python run_experiments.py --dataset cora --output-dir my_experiment_results
```

### 2. Custom random seed for reproducibility
```bash
python run_experiments.py --dataset cora --random-seed 123
```

### 3. Custom dataset parameters
```bash
python run_experiments.py --dataset cora --k 5 --percent-positive 0.2
```

### 4. Combine multiple customizations
```bash
python run_experiments.py \
  --dataset cora citeseer \
  --num-particles 100 200 387 \
  --cluster-strategy percentage \
  --n-runs 5 \
  --output-dir comprehensive_results \
  --quick-test
```

## 📊 Output Files

For each dataset, the script generates:

1. **`{dataset}_final_results.csv`** - All experiment results including:
   - All parameter values
   - F1 score, precision, recall
   - Coverage percentage
   - Number of reliable negatives
   - Graph statistics
   - Run status

2. **`{dataset}_summary_results.csv`** - Aggregated results by parameter combination:
   - Mean, std, min, max for F1 score
   - Mean, std for precision, recall
   - Mean, std for coverage and reliable negatives

## 🔍 Example Workflows

### Quick Testing Workflow
```bash
# 1. Quick test to verify everything works
python run_experiments.py --dataset cora --quick-test --n-runs 2

# 2. If successful, run full experiments
python run_experiments.py --dataset cora --n-runs 5
```

### Parameter Tuning Workflow
```bash
# 1. Test broad parameter ranges
python run_experiments.py --dataset cora --num-particles 50 100 200 500

# 2. Focus on promising ranges
python run_experiments.py --dataset cora --num-particles 100 150 200 250

# 3. Fine-tune best parameters
python run_experiments.py --dataset cora --num-particles 180 190 200 210
```

### Multi-Dataset Comparison
```bash
# Run same parameters on multiple datasets
python run_experiments.py \
  --dataset cora citeseer twitch \
  --num-particles 100 200 387 \
  --n-runs 3
```

## 📝 Parameter Descriptions

| Parameter | Description | Default Values |
|-----------|-------------|----------------|
| `--dataset` | Dataset(s) to test | `cora` |
| `--n-runs` | Runs per combination | `3` |
| `--output-dir` | Results directory | `experiment_results` |
| `--random-seed` | Random seed | `42` |
| `--quick-test` | Use reduced parameters | `False` |
| `--num-particles` | Number of particles | `[50, 100, 200, 387, 500]` |
| `--cluster-strategy` | Cluster labeling | `[majority, percentage]` |
| `--positive-cluster-threshold` | Positive threshold | `[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9]` |
| `--movement-strategy` | Particle movement | `[uniform, degree_weighted]` |
| `--initialization-strategy` | Particle init | `[random, degree_weighted]` |
| `--avg-node-pot-threshold` | Stopping threshold | `[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]` |

## ⚠️ Important Notes

1. **`p_det`, `delta_v`, and `delta_p` are fixed constants** (0.6, 0.3, 0.4) hardcoded in the model
2. **Coverage is automatically tracked** and included in all results
3. **Intermediate results are saved** every 10 experiments to prevent data loss
4. **Error handling** continues experiments even if some fail
5. **Progress tracking** shows current status and estimated completion time

## 🚨 Troubleshooting

### Common Issues:
- **Memory errors**: Use `--quick-test` or reduce parameter ranges
- **Long runtime**: Reduce `--n-runs` or use `--quick-test`
- **Import errors**: Ensure virtual environment is activated
- **Permission errors**: Check output directory permissions

### Performance Tips:
- Start with `--quick-test` to verify setup
- Use smaller `--n-runs` for initial testing
- Focus on one dataset at a time
- Monitor system resources during execution 