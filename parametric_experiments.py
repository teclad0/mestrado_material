import networkx as nx
import pandas as pd
import numpy as np
import random
import json
import os
import time
from typing import Dict, List, Any, Tuple, Optional
from itertools import product
from tqdm import tqdm
import warnings
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

from model import PULearningPC, ReliableNegativeSelection
from generate_dataset import (
    load_cora_scar, load_citeseer_scar, load_pubmed_scar, load_twitch_scar,
    load_mnist_scar, load_amazon_photo_scar, relabel_graph_scar
)
from dataset_loader import DatasetLoader, load_dataset_for_model
from models_experiment import evaluate_reliable_negative_metrics, evaluate_phase2, classify_with_svm
from experiment_config import PARAMETER_RANGES, DATASET_CONFIG

class PULearningExperimentRunner:
    """
    Comprehensive parametric experiment runner for PULearningPC algorithm.
    Tests multiple parameter combinations with multiple runs per combination.
    """
    
    def __init__(self, 
                 n_runs: int = 5,
                 output_dir: str = "experiment_results",
                 random_seed: int = 42,
                 dataset_name: str = None,
                 use_saved_datasets: bool = False,
                 datasets_dir: str = "datasets"):
        """
        Initialize the experiment runner.
        
        Args:
            n_runs: Number of runs per parameter combination
            output_dir: Directory to save results
            random_seed: Random seed for reproducibility
            dataset_name: Name of the dataset (for backward compatibility)
            use_saved_datasets: Whether to use pre-generated datasets
            datasets_dir: Directory containing pre-generated datasets
        """
        self.n_runs = n_runs
        self.output_dir = output_dir
        self.random_seed = random_seed
        self.use_saved_datasets = use_saved_datasets
        self.datasets_dir = datasets_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize dataset loader if using saved datasets
        if use_saved_datasets:
            self.dataset_loader = DatasetLoader(datasets_dir)
        
        # Initialize results storage
        self.results = []
        
    def get_parameter_grid(self) -> List[Dict[str, Any]]:
        """
        Define the parameter grid for experiments.
        Returns a list of parameter dictionaries to test.
        """
        # Check if custom parameters were set
        if hasattr(self, '_custom_parameter_ranges'):
            param_ranges = self._custom_parameter_ranges
        else:
            param_ranges = PARAMETER_RANGES

        # Generate all combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())

        combinations = []
        for values in product(*param_values):
            param_dict = dict(zip(param_names, values))
            combinations.append(param_dict)

        return combinations

    def get_simulation_and_rns_grids(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split the parameter grid into expensive simulation params and cheap RNS params.

        Simulation params: num_particles, movement_strategy, initialization_strategy, avg_node_pot_threshold
        RNS params: cluster_strategy, positive_cluster_threshold

        Also eliminates redundant majority+threshold combinations (majority ignores threshold).

        Returns:
            Tuple of (simulation_combinations, rns_combinations)
        """
        if hasattr(self, '_custom_parameter_ranges'):
            param_ranges = self._custom_parameter_ranges
        else:
            param_ranges = PARAMETER_RANGES

        # Simulation parameters (expensive — control particle competition)
        sim_param_names = ['num_particles', 'movement_strategy', 'initialization_strategy', 'avg_node_pot_threshold']
        sim_values = [param_ranges[k] for k in sim_param_names if k in param_ranges]
        sim_names = [k for k in sim_param_names if k in param_ranges]

        sim_combinations = []
        for values in product(*sim_values):
            sim_combinations.append(dict(zip(sim_names, values)))

        # RNS parameters (cheap — only post-processing)
        # For 'majority' strategy, threshold is irrelevant, so include only once
        rns_combinations = []
        cluster_strategies = param_ranges.get('cluster_strategy', ['percentage'])
        thresholds = param_ranges.get('positive_cluster_threshold', [0.5])

        for strategy in cluster_strategies:
            if strategy == 'majority':
                rns_combinations.append({
                    'cluster_strategy': 'majority',
                    'positive_cluster_threshold': thresholds[0]  # placeholder, ignored
                })
            else:
                for thresh in thresholds:
                    rns_combinations.append({
                        'cluster_strategy': strategy,
                        'positive_cluster_threshold': thresh
                    })

        return sim_combinations, rns_combinations
    
    def set_custom_parameter_ranges(self, param_ranges: Dict[str, List[Any]]):
        """
        Set custom parameter ranges for experiments.
        
        Args:
            param_ranges: Dictionary with parameter names as keys and lists of values as values
        """
        self._custom_parameter_ranges = param_ranges
    
    def get_current_parameter_ranges(self) -> Dict[str, List[Any]]:
        """
        Get the current parameter ranges being used.

        Returns:
            Dictionary with current parameter ranges
        """
        if hasattr(self, '_custom_parameter_ranges'):
            return self._custom_parameter_ranges
        else:
            return PARAMETER_RANGES
    
    def load_dataset(self, dataset_name: str, dataset_filename: str = None, **kwargs) -> nx.Graph:
        """
        Load a specific dataset with default parameters.
        
        Args:
            dataset_name: Name of the dataset to load
            dataset_filename: Optional filename for pre-generated dataset
            **kwargs: Additional parameters for dataset loading
            
        Returns:
            NetworkX graph
        """
        if self.use_saved_datasets and dataset_filename:
            # Load from pre-generated dataset
            return self.dataset_loader.load_dataset_as_networkx(dataset_filename)

        # Use centralized dataset config
        default_params = DATASET_CONFIG

        # Update with provided kwargs
        params = default_params.get(dataset_name, {}).copy()
        params.update(kwargs)

        # Load dataset
        if dataset_name == 'cora':
            return load_cora_scar(**params)
        elif dataset_name == 'citeseer':
            return load_citeseer_scar(**params)
        elif dataset_name == 'pubmed':
            return load_pubmed_scar(**params)
        elif dataset_name == 'twitch':
            return load_twitch_scar(**params)
        elif dataset_name == 'mnist':
            return load_mnist_scar(**params)
        elif dataset_name == 'amazon_photo':
            return load_amazon_photo_scar(**params)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    

    def run_experiments(self,
                       dataset_name: str,
                       dataset_kwargs: Dict[str, Any] = None,
                       dataset_filename: str = None,
                       n_jobs: int = None,
                       num_neg: int = 200) -> pd.DataFrame:
        """
        Run all experiments for a given dataset.

        Uses an optimized strategy: simulation parameters (expensive) are separated
        from RNS parameters (cheap). Each simulation runs once, then multiple RNS
        configurations are evaluated on the cached result.

        Args:
            dataset_name: Name of the dataset to use
            dataset_kwargs: Additional parameters for dataset loading
            dataset_filename: Optional filename for pre-generated dataset
            n_jobs: Number of parallel jobs (None = use all available cores)
            num_neg: Number of reliable negatives to select (default: 200)

        Returns:
            DataFrame with all experiment results
        """
        print(f"Starting experiments on {dataset_name} dataset...")

        # Split grid into simulation (expensive) and RNS (cheap) params
        sim_combinations, rns_combinations = self.get_simulation_and_rns_grids()

        current_ranges = self.get_current_parameter_ranges()
        total_simulations = len(sim_combinations) * self.n_runs
        total_evaluations = total_simulations * len(rns_combinations)

        print(f"Parameter ranges being used:")
        for param, values in current_ranges.items():
            print(f"  {param}: {values}")

        print(f"\nOptimized execution plan:")
        print(f"  Unique simulation configs: {len(sim_combinations)}")
        print(f"  RNS configs per simulation: {len(rns_combinations)} (majority×1 + percentage×thresholds)")
        print(f"  Runs per simulation config: {self.n_runs}")
        print(f"  Total expensive simulations: {total_simulations}")
        print(f"  Total evaluations (results rows): {total_evaluations}")

        naive_total = len(self.get_parameter_grid()) * self.n_runs
        percent_positive_values = current_ranges.get('percent_positive', None)
        if percent_positive_values:
            naive_total *= 1  # already in grid
            total_evaluations *= len(percent_positive_values)
            total_simulations *= len(percent_positive_values)
        savings = (1 - total_simulations / naive_total) * 100 if naive_total > 0 else 0
        print(f"  Simulation savings vs naive grid: {savings:.0f}% fewer simulations")

        # Determine number of parallel jobs
        if n_jobs is None:
            n_jobs = min(mp.cpu_count(), 8)
        print(f"  Using {n_jobs} parallel processes")

        # Group experiments by percent_positive (each value needs different SCAR labeling)
        if percent_positive_values:
            # Build graph ONCE (structure + features + true_label), then relabel for each pct
            print(f"\n--- Building base graph for {dataset_name} (structure reused across percent_positive values) ---")
            kwargs = (dataset_kwargs or {}).copy()
            kwargs['percent_positive'] = percent_positive_values[0]  # initial load
            base_graph = self.load_dataset(dataset_name, dataset_filename, **kwargs)
            print(f"Base graph: {base_graph.number_of_nodes()} nodes, {base_graph.number_of_edges()} edges")

            all_results = []
            for pct_pos in percent_positive_values:
                print(f"\n--- Re-labeling with percent_positive={pct_pos} ---")
                graph = relabel_graph_scar(base_graph, pct_pos)

                results = self._run_experiments_optimized(
                    graph, sim_combinations, rns_combinations, n_jobs, num_neg, dataset_name, pct_pos
                )
                all_results.extend(results)

            df = pd.DataFrame(all_results)
        else:
            # No percent_positive in grid — load dataset once with kwargs
            graph = self.load_dataset(dataset_name, dataset_filename, **(dataset_kwargs or {}))
            print(f"Loaded {dataset_name}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            results = self._run_experiments_optimized(
                graph, sim_combinations, rns_combinations, n_jobs, num_neg, dataset_name, None
            )
            df = pd.DataFrame(results)

        # Save final results
        self.save_final_results(df, dataset_name)

        return df
    
    @staticmethod
    def _format_params_short(params: Dict[str, Any]) -> str:
        """Format parameters as a compact string for logging."""
        pct = params.get('percent_positive', None)
        pct_str = f"pct={pct}, " if pct is not None else ""
        return (f"{pct_str}p={params['num_particles']}, "
                f"strat={params['cluster_strategy']}, "
                f"thresh={params['positive_cluster_threshold']}, "
                f"mov={params['movement_strategy']}, "
                f"init={params['initialization_strategy']}, "
                f"avg_thresh={params['avg_node_pot_threshold']}")

    def _run_experiments_optimized(self,
                                  graph: nx.Graph,
                                  sim_combinations: List[Dict[str, Any]],
                                  rns_combinations: List[Dict[str, Any]],
                                  n_jobs: int,
                                  num_neg: int,
                                  dataset_name: str,
                                  percent_positive: Optional[float]) -> List[Dict[str, Any]]:
        """
        Optimized experiment execution: run each simulation config once,
        then evaluate multiple RNS configs on the cached result.
        """
        # Build tasks: one per (sim_config, run_id)
        all_tasks = []
        for i, sim_params in enumerate(sim_combinations):
            for run_id in range(self.n_runs):
                all_tasks.append((sim_params, run_id, i))

        total_simulations = len(all_tasks)
        total_evaluations = total_simulations * len(rns_combinations)
        print(f"Running {total_simulations} simulations → {total_evaluations} total evaluations")

        run_worker = partial(
            self._run_simulation_and_evaluate_rns,
            graph, num_neg, rns_combinations, percent_positive
        )

        results = []
        completed = 0
        successful = 0
        failed = 0
        task_times = []
        experiment_start = time.time()

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_task = {}
            submit_times = {}

            for task_idx, (sim_params, run_id, i) in enumerate(all_tasks):
                future = executor.submit(run_worker, sim_params, run_id, task_idx)
                future_to_task[future] = (sim_params, run_id, task_idx)
                submit_times[future] = time.time()

            with tqdm(total=total_simulations, desc="Running simulations") as pbar:
                for future in as_completed(future_to_task):
                    sim_params, run_id, task_idx = future_to_task[future]
                    elapsed = time.time() - submit_times[future]

                    try:
                        batch_results = future.result()
                        results.extend(batch_results)
                        completed += 1
                        pbar.update(1)

                        # Track success/failure
                        batch_successes = sum(1 for r in batch_results if r.get('status') == 'success')
                        batch_failures = len(batch_results) - batch_successes
                        if batch_successes > 0:
                            successful += 1
                        else:
                            failed += 1
                        task_times.append(elapsed)

                        # Log result with final_avg_potential
                        first = batch_results[0] if batch_results else {}
                        status = first.get('status', 'unknown')
                        iters = first.get('final_iteration_count', 0)
                        coverage = first.get('coverage', 0.0)
                        avg_thresh_reached = first.get('avg_threshold_criteria', False)

                        sim_str = (
                            f"p={sim_params.get('num_particles')}, "
                            f"mov={sim_params.get('movement_strategy')}, "
                            f"init={sim_params.get('initialization_strategy')}, "
                            f"avg_thresh={sim_params.get('avg_node_pot_threshold')}"
                        )
                        best_f1 = max((r.get('step1_f1_score', 0) for r in batch_results), default=0)
                        tqdm.write(
                            f"  [{completed}/{total_simulations}] {elapsed:.1f}s | "
                            f"{dataset_name} | run={run_id} | {sim_str} | "
                            f"iters={iters}, cov={coverage:.2f}, thresh_reached={avg_thresh_reached}, "
                            f"best_F1={best_f1:.3f}, status={status}"
                        )

                        # Write live status JSON
                        now = time.time()
                        in_flight_count = sum(1 for f in future_to_task if not f.done())
                        avg_time = np.mean(task_times) if task_times else 0
                        remaining_tasks = total_simulations - completed
                        eta_seconds = avg_time * remaining_tasks / max(n_jobs, 1)

                        self._write_live_status(
                            dataset_name=dataset_name,
                            total_tasks=total_simulations,
                            completed=completed,
                            successful=successful,
                            failed=failed,
                            in_flight=in_flight_count,
                            elapsed_seconds=now - experiment_start,
                            avg_task_time=avg_time,
                            eta_seconds=eta_seconds,
                            percent_positive=percent_positive,
                            results=results,
                            recent_task={
                                'task_idx': task_idx,
                                'params': sim_str,
                                'time': round(elapsed, 1),
                                'f1': round(best_f1, 4),
                                'coverage': round(coverage, 4),
                                'iters': iters,
                                'status': status
                            }
                        )

                        if completed % 20 == 0:
                            self.save_intermediate_results(results, f"{dataset_name}_intermediate")

                    except Exception as e:
                        tqdm.write(f"  ERROR for task {task_idx} after {elapsed:.1f}s: {e}")
                        # Generate error results for all RNS combos
                        for rns_params in rns_combinations:
                            error_result = {
                                'run_id': run_id,
                                'percent_positive': percent_positive,
                                'num_particles': sim_params.get('num_particles', -1),
                                'cluster_strategy': rns_params.get('cluster_strategy', 'error'),
                                'positive_cluster_threshold': rns_params.get('positive_cluster_threshold', -1),
                                'movement_strategy': sim_params.get('movement_strategy', 'error'),
                                'initialization_strategy': sim_params.get('initialization_strategy', 'error'),
                                'avg_node_pot_threshold': sim_params.get('avg_node_pot_threshold', -1),
                                'step1_f1_score': 0.0,
                                'step1_precision': 0.0,
                                'step1_recall': 0.0,
                                'step1_num_reliable_negatives': 0,
                                'step2_f1': 0.0,
                                'step2_precision': 0.0,
                                'step2_recall': 0.0,
                                'step2_accuracy': 0.0,
                                'coverage': 0.0,
                                'fallback_rule_used': False,
                                'avg_threshold_criteria': False,
                                'final_iteration_count': 0,
                                'graph_nodes': graph.number_of_nodes() if graph else 0,
                                'graph_edges': graph.number_of_edges() if graph else 0,
                                'status': f'error: {str(e)}'
                            }
                            results.append(error_result)
                        completed += 1
                        failed += 1
                        task_times.append(elapsed)
                        pbar.update(1)

        return results
    
    def _run_simulation_and_evaluate_rns(self,
                                        graph: nx.Graph,
                                        num_neg: int,
                                        rns_combinations: List[Dict[str, Any]],
                                        percent_positive: Optional[float],
                                        sim_params: Dict[str, Any],
                                        run_id: int,
                                        task_idx: int) -> List[Dict[str, Any]]:
        """
        Worker: run ONE simulation, then evaluate ALL RNS configs on the result.

        Returns a list of result dicts (one per RNS config).
        """
        results = []

        try:
            pcm_params = {
                'num_particles': sim_params['num_particles'],
                'movement_strategy': sim_params['movement_strategy'],
                'initialization_strategy': sim_params['initialization_strategy'],
                'average_node_potential_threshold': sim_params['avg_node_pot_threshold']
            }

            # Run simulation ONCE (the expensive part)
            model = PULearningPC(
                graph=graph,
                num_neg=num_neg,
                pcm_params=pcm_params,
                rns_params={'cluster_strategy': 'percentage', 'positive_cluster_threshold': 0.1}
            )
            model.train()

            # Cache simulation state
            final_coverage = model.pcm.get_graph_coverage()[1]
            avg_threshold_criteria = model.labeled_graph.graph.get('avg_threshold_criteria', False)
            final_iteration_count = model.labeled_graph.graph.get('final_iteration_count', 0)

            # Evaluate each RNS configuration on the cached simulation (the cheap part)
            for rns_params in rns_combinations:
                try:
                    rns = ReliableNegativeSelection(
                        model.labeled_graph,
                        owner_groups=model.pcm.owner_groups,
                        cluster_sizes=model.pcm.cluster_sizes,
                        cluster_positive_counts=model.pcm.cluster_positive_counts,
                        **rns_params
                    )
                    rns.assign_cluster_label()
                    rns.calculate_dissimilarity()
                    ranked_nodes = rns.rank_nodes_dissimilarity(num_neg=num_neg)

                    fallback_rule_used = rns.graph.graph.get('fallback_rule_used', False)

                    if not ranked_nodes:
                        results.append({
                            'run_id': run_id,
                            'percent_positive': percent_positive,
                            'num_particles': sim_params['num_particles'],
                            'cluster_strategy': rns_params['cluster_strategy'],
                            'positive_cluster_threshold': rns_params['positive_cluster_threshold'],
                            'movement_strategy': sim_params['movement_strategy'],
                            'initialization_strategy': sim_params['initialization_strategy'],
                            'avg_node_pot_threshold': sim_params['avg_node_pot_threshold'],
                            'step1_f1_score': 0.0,
                            'step1_precision': 0.0,
                            'step1_recall': 0.0,
                            'step1_num_reliable_negatives': 0,
                            'step2_f1': 0.0,
                            'step2_precision': 0.0,
                            'step2_recall': 0.0,
                            'step2_accuracy': 0.0,
                            'coverage': final_coverage,
                            'fallback_rule_used': fallback_rule_used,
                            'avg_threshold_criteria': avg_threshold_criteria,
                            'final_iteration_count': final_iteration_count,
                            'graph_nodes': graph.number_of_nodes(),
                            'graph_edges': graph.number_of_edges(),
                            'status': 'error: insufficient reliable negatives'
                        })
                        continue

                    reliable_negatives = [node for node, _ in ranked_nodes]

                    # Step 1 metrics
                    metrics = evaluate_reliable_negative_metrics(graph, reliable_negatives)

                    # Step 2: SVM classification
                    predictions = classify_with_svm(graph, reliable_negatives)
                    step2_metrics = evaluate_phase2(graph, predictions)

                    results.append({
                        'run_id': run_id,
                        'percent_positive': percent_positive,
                        'num_particles': sim_params['num_particles'],
                        'cluster_strategy': rns_params['cluster_strategy'],
                        'positive_cluster_threshold': rns_params['positive_cluster_threshold'],
                        'movement_strategy': sim_params['movement_strategy'],
                        'initialization_strategy': sim_params['initialization_strategy'],
                        'avg_node_pot_threshold': sim_params['avg_node_pot_threshold'],
                        'step1_f1_score': metrics['f1_score'],
                        'step1_precision': metrics['precision'],
                        'step1_recall': metrics['recall'],
                        'step1_num_reliable_negatives': metrics['num_reliable_negatives'],
                        'step2_f1': step2_metrics['f1'],
                        'step2_precision': step2_metrics['precision'],
                        'step2_recall': step2_metrics['recall'],
                        'step2_accuracy': step2_metrics['accuracy'],
                        'coverage': final_coverage,
                        'fallback_rule_used': fallback_rule_used,
                        'avg_threshold_criteria': avg_threshold_criteria,
                        'final_iteration_count': final_iteration_count,
                        'graph_nodes': graph.number_of_nodes(),
                        'graph_edges': graph.number_of_edges(),
                        'status': 'success'
                    })

                except Exception as e:
                    results.append({
                        'run_id': run_id,
                        'percent_positive': percent_positive,
                        'num_particles': sim_params['num_particles'],
                        'cluster_strategy': rns_params.get('cluster_strategy', 'error'),
                        'positive_cluster_threshold': rns_params.get('positive_cluster_threshold', -1),
                        'movement_strategy': sim_params['movement_strategy'],
                        'initialization_strategy': sim_params['initialization_strategy'],
                        'avg_node_pot_threshold': sim_params['avg_node_pot_threshold'],
                        'step1_f1_score': 0.0,
                        'step1_precision': 0.0,
                        'step1_recall': 0.0,
                        'step1_num_reliable_negatives': 0,
                        'step2_f1': 0.0,
                        'step2_precision': 0.0,
                        'step2_recall': 0.0,
                        'step2_accuracy': 0.0,
                        'coverage': final_coverage,
                        'fallback_rule_used': False,
                        'avg_threshold_criteria': avg_threshold_criteria,
                        'final_iteration_count': final_iteration_count,
                        'graph_nodes': graph.number_of_nodes(),
                        'graph_edges': graph.number_of_edges(),
                        'status': f'error: {str(e)}'
                    })

        except Exception as e:
            # Simulation itself failed — generate error for all RNS combos
            for rns_params in rns_combinations:
                results.append({
                    'run_id': run_id,
                    'percent_positive': percent_positive,
                    'num_particles': sim_params.get('num_particles', -1),
                    'cluster_strategy': rns_params.get('cluster_strategy', 'error'),
                    'positive_cluster_threshold': rns_params.get('positive_cluster_threshold', -1),
                    'movement_strategy': sim_params.get('movement_strategy', 'error'),
                    'initialization_strategy': sim_params.get('initialization_strategy', 'error'),
                    'avg_node_pot_threshold': sim_params.get('avg_node_pot_threshold', -1),
                    'step1_f1_score': 0.0,
                    'step1_precision': 0.0,
                    'step1_recall': 0.0,
                    'step1_num_reliable_negatives': 0,
                    'step2_f1': 0.0,
                    'step2_precision': 0.0,
                    'step2_recall': 0.0,
                    'step2_accuracy': 0.0,
                    'coverage': 0.0,
                    'fallback_rule_used': False,
                    'avg_threshold_criteria': False,
                    'final_iteration_count': 0,
                    'graph_nodes': graph.number_of_nodes() if graph else 0,
                    'graph_edges': graph.number_of_edges() if graph else 0,
                    'status': f'error: {str(e)}'
                })

        return results

    def _write_live_status(self, dataset_name: str, total_tasks: int, completed: int,
                           successful: int, failed: int, in_flight: int,
                           elapsed_seconds: float, avg_task_time: float,
                           eta_seconds: float, percent_positive: Optional[float],
                           results: List[Dict], recent_task: Dict):
        """Write live status JSON for the dashboard."""
        import datetime

        # Aggregate by num_particles
        completed_by_particles = {}
        best_f1_by_particles = {}
        best_acc_by_particles = {}
        for r in results:
            p = str(r.get('num_particles', '?'))
            if p not in completed_by_particles:
                completed_by_particles[p] = 0
                best_f1_by_particles[p] = 0.0
                best_acc_by_particles[p] = 0.0
            completed_by_particles[p] += 1
            if r.get('status') == 'success':
                best_f1_by_particles[p] = max(best_f1_by_particles[p], r.get('step1_f1_score', 0.0))
                best_acc_by_particles[p] = max(best_acc_by_particles[p], r.get('step2_accuracy', 0.0))

        # Maintain recent tasks list (last 15)
        status_path = f"{self.output_dir}/live_status.json"
        recent_tasks = []
        try:
            with open(status_path, 'r') as f:
                old = json.load(f)
                recent_tasks = old.get('recent_tasks', [])
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        recent_tasks.append(recent_task)
        recent_tasks = recent_tasks[-15:]

        status = {
            'dataset': dataset_name,
            'percent_positive': percent_positive,
            'total_tasks': total_tasks,
            'completed': completed,
            'successful': successful,
            'failed': failed,
            'in_flight': in_flight,
            'elapsed_seconds': round(elapsed_seconds, 1),
            'avg_task_time': round(avg_task_time, 1),
            'eta_seconds': round(eta_seconds, 1),
            'last_updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'completed_by_particles': completed_by_particles,
            'best_f1_by_particles': best_f1_by_particles,
            'best_accuracy_by_particles': best_acc_by_particles,
            'recent_tasks': recent_tasks
        }

        try:
            with open(status_path, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception:
            pass

    def save_intermediate_results(self, results: List[Dict], dataset_name: str):
        """Save intermediate results to avoid losing progress."""
        df = pd.DataFrame(results)
        filename = f"{self.output_dir}/{dataset_name}_intermediate_results.csv"
        df.to_csv(filename, index=False)
    
    def save_final_results(self, df: pd.DataFrame, dataset_name: str):
        """Save final results to CSV."""
        filename = f"{self.output_dir}/{dataset_name}_final_results.csv"
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        
        # Also save a summary
        agg_dict = {
            'step1_f1_score': ['mean', 'std', 'min', 'max'],
            'step1_precision': ['mean', 'std'],
            'step1_recall': ['mean', 'std'],
            'step1_num_reliable_negatives': ['mean', 'std'],
            'coverage': ['mean', 'std', 'min', 'max'],
            'fallback_rule_used': ['sum', 'count'],
            'avg_threshold_criteria': ['sum', 'count'],
            'final_iteration_count': ['mean', 'std', 'min', 'max']
        }
        if 'step2_f1' in df.columns:
            agg_dict['step2_f1'] = ['mean', 'std', 'min', 'max']
            agg_dict['step2_precision'] = ['mean', 'std']
            agg_dict['step2_recall'] = ['mean', 'std']
            agg_dict['step2_accuracy'] = ['mean', 'std']

        groupby_cols = [
            'num_particles', 'cluster_strategy',
            'positive_cluster_threshold', 'movement_strategy', 'initialization_strategy',
            'avg_node_pot_threshold'
        ]
        if 'percent_positive' in df.columns and df['percent_positive'].notna().any():
            groupby_cols.insert(0, 'percent_positive')
        summary = df.groupby(groupby_cols).agg(agg_dict).round(4)
        
        summary_filename = f"{self.output_dir}/{dataset_name}_summary_results.csv"
        summary.to_csv(summary_filename)
        print(f"Summary saved to {summary_filename}")
        
        # Print some key iteration statistics
        if 'final_iteration_count' in df.columns:
            print(f"\nIteration Statistics for {dataset_name}:")
            print(f"  Average iterations: {df['final_iteration_count'].mean():.1f}")
            print(f"  Min iterations: {df['final_iteration_count'].min()}")
            print(f"  Max iterations: {df['final_iteration_count'].max()}")
            print(f"  Std iterations: {df['final_iteration_count'].std():.1f}")
    
    def run_all_datasets(self):
        """Run experiments on all available datasets."""
        datasets = ['cora', 'citeseer', 'twitch', 'mnist']
        
        all_results = {}
        for dataset in datasets:
            try:
                print(f"\n{'='*50}")
                print(f"Running experiments on {dataset.upper()} dataset")
                print(f"{'='*50}")
                
                results_df = self.run_experiments(dataset)
                all_results[dataset] = results_df
                
            except Exception as e:
                print(f"Error running experiments on {dataset}: {e}")
                continue
        
        return all_results

def main():
    """Main function to run experiments."""
    # Initialize experiment runner
    runner = PULearningExperimentRunner(
        n_runs=5,  # Number of runs per parameter combination
        output_dir="experiment_results",
        random_seed=42
    )
    
    # Run experiments on a specific dataset
    print("Running experiments on Cora dataset...")
    cora_results = runner.run_experiments('cora')
    
    # Or run on all datasets
    # all_results = runner.run_all_datasets()
    
    print("\nExperiments completed!")
    print(f"Results saved in: {runner.output_dir}/")

if __name__ == "__main__":
    main() 