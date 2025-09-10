# Updated Analysis Script

The `analyze_results.py` script has been updated with new functionality to support both threshold filtering and aggregation across all thresholds.

## New Features

### Filter by Threshold Option

The script now supports a `filter_by_threshold` parameter that controls how the analysis is performed:

- **`filter_by_threshold=True` (default)**: Filters data by a specific `positive_cluster_threshold` value (original behavior)
- **`filter_by_threshold=False`**: Aggregates data across all `positive_cluster_threshold` values, taking the mean F1 score

### Updated Methods

All chart generation methods now support the new parameter:

- `generate_f1_vs_particles_charts()`
- `generate_heatmap_charts()`
- `generate_parameter_comparison_chart()`
- `generate_statistical_summary()`
- `run_full_analysis()`

### Command Line Usage

```bash
# Original behavior (filter by threshold=0.1)
python analyze_results.py --threshold 0.1

# New behavior (aggregate across all thresholds)
python analyze_results.py --no-filter-by-threshold

# Explicit filtering (same as original)
python analyze_results.py --filter-by-threshold --threshold 0.1
```

### Programmatic Usage

```python
from analyze_results import ExperimentResultsAnalyzer

analyzer = ExperimentResultsAnalyzer("experiment_results")
analyzer.load_summary_results()

# Filter by threshold (original behavior)
analyzer.generate_f1_vs_particles_charts(
    threshold=0.1,
    filter_by_threshold=True,
    save_dir="charts_filtered"
)

# Aggregate across all thresholds (new behavior)
analyzer.generate_f1_vs_particles_charts(
    threshold=0.1,  # Ignored when filter_by_threshold=False
    filter_by_threshold=False,
    save_dir="charts_aggregated"
)
```

## Output Files

When `filter_by_threshold=False`, the output files are renamed to indicate aggregation:

- `f1_vs_particles_init_{strategy}_move_{strategy}_aggregated.png`
- `heatmap_init_{strategy}_move_{strategy}_aggregated.png`
- `comprehensive_f1_vs_particles_aggregated.png`
- `statistical_summary_aggregated.csv`

## Use Cases

### Threshold Filtering (Original)
- Analyze performance at a specific threshold value
- Compare different parameter combinations at a fixed threshold
- Generate heatmaps showing F1 vs particles vs avg_node_pot_threshold

### Aggregation (New)
- Analyze overall performance across all threshold values
- Get a general view of how F1 scores vary with number of particles
- Reduce noise by averaging across threshold variations
- Focus on the core relationship between particles and performance

## Example

See `example_usage_updated.py` for a complete example demonstrating both modes.
