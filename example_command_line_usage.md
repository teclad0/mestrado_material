# Command Line Usage Examples for PULearningPC Experiments

This document shows how to use the `run_experiments.py` script to run parametric experiments from the command line.

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

### 2. Test specific delta values
```bash
python run_experiments.py --dataset cora --delta-v 0.2 0.3 0.4 --delta-p 0.5 0.7 0.8
```

### 3. Test specific cluster strategies
```bash
python run_experiments.py --dataset cora --cluster-strategy majority percentage
```

### 4. Test specific thresholds
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
  --delta-v 0.2 0.3 0.4 \
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
  --delta-v 0.3 \
  --delta-p 0.4 \
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
| `--delta-v` | Velocity decay | `[0.1, 0.2, 0.3, 0.4, 0.5]` |
| `--delta-p` | Potential decay | `[0.3, 0.5, 0.7, 0.8, 0.9]` |
| `--cluster-strategy` | Cluster labeling | `[majority, percentage]` |
| `--positive-cluster-threshold` | Positive threshold | `[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9]` |
| `--movement-strategy` | Particle movement | `[uniform, degree_weighted]` |
| `--initialization-strategy` | Particle init | `[random, degree_weighted]` |
| `--avg-node-pot-threshold` | Stopping threshold | `[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]` |

## ⚠️ Important Notes

1. **`p_det` is fixed at 0.6** as requested in your requirements
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