import networkx as nx
import pandas as pd
import numpy as np
import random
import json
import os
import time
from typing import Dict, List, Any, Tuple
from itertools import product
from tqdm import tqdm
import warnings
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

from model import PULearningPC
from generate_dataset import (
    load_cora_scar, load_citeseer_scar, load_pubmed_scar, load_twitch_scar,
    load_mnist_scar
)
from dataset_loader import DatasetLoader, load_dataset_for_model
from models_experiment import evaluate_reliable_negative_metrics, evaluate_phase2
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
        
        # Load dataset
        graph = self.load_dataset(dataset_name, dataset_filename, **(dataset_kwargs or {}))
        print(f"Loaded {dataset_name}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        # Get parameter combinations
        param_combinations = self.get_parameter_grid()
        total_experiments = len(param_combinations) * self.n_runs
        
        print(f"Running {total_experiments} experiments ({len(param_combinations)} parameter combinations × {self.n_runs} runs each)")
        
        # Show which parameter ranges are being used
        current_ranges = self.get_current_parameter_ranges()
        print(f"Parameter ranges being used:")
        for param, values in current_ranges.items():
            print(f"  {param}: {values}")
        
        # Determine number of parallel jobs
        if n_jobs is None:
            n_jobs = min(mp.cpu_count(), 8)  # Use all cores but cap at 8 to avoid memory issues
        
        print(f"Using {n_jobs} parallel processes")
        
        # Run experiments in parallel
        results = self._run_experiments_parallel(graph, param_combinations, n_jobs, num_neg, dataset_name)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save final results
        self.save_final_results(df, dataset_name)
        
        return df
    
    @staticmethod
    def _format_params_short(params: Dict[str, Any]) -> str:
        """Format parameters as a compact string for logging."""
        return (f"p={params['num_particles']}, "
                f"strat={params['cluster_strategy']}, "
                f"thresh={params['positive_cluster_threshold']}, "
                f"mov={params['movement_strategy']}, "
                f"init={params['initialization_strategy']}, "
                f"avg_thresh={params['avg_node_pot_threshold']}")

    def _run_experiments_parallel(self,
                                 graph: nx.Graph,
                                 param_combinations: List[Dict[str, Any]],
                                 n_jobs: int,
                                 num_neg: int,
                                 dataset_name: str) -> List[Dict[str, Any]]:
        """
        Run experiments in parallel using multiprocessing.
        """
        all_tasks = []
        for i, params in enumerate(param_combinations):
            for run_id in range(self.n_runs):
                all_tasks.append((params, run_id, i))

        total_tasks = len(all_tasks)
        print(f"Total tasks to process: {total_tasks}")

        run_single_experiment_partial = partial(self._run_single_experiment_worker, graph, num_neg)

        results = []
        completed = 0

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_task = {}
            submit_times = {}

            for task_idx, (params, run_id, i) in enumerate(all_tasks):
                future = executor.submit(run_single_experiment_partial, params, run_id, task_idx)
                future_to_task[future] = (params, run_id, task_idx)
                submit_times[future] = time.time()

            with tqdm(total=total_tasks, desc="Running experiments") as pbar:
                for future in as_completed(future_to_task):
                    params, run_id, task_idx = future_to_task[future]
                    elapsed = time.time() - submit_times[future]

                    try:
                        result = future.result()
                        results.append(result)
                        completed += 1
                        pbar.update(1)

                        param_str = self._format_params_short(params)
                        status = result.get('status', 'unknown')
                        iters = result.get('final_iteration_count', 0)
                        s1_f1 = result.get('step1_f1_score', 0.0)
                        s2_f1 = result.get('step2_f1', 0.0)

                        tqdm.write(
                            f"  [{completed}/{total_tasks}] DONE in {elapsed:.1f}s | "
                            f"{dataset_name} | run={run_id} | {param_str} | "
                            f"iters={iters}, S1_F1={s1_f1:.3f}, S2_F1={s2_f1:.3f}, status={status}"
                        )

                        # Show in-flight tasks sorted by elapsed time
                        in_flight = []
                        now = time.time()
                        for f, (p, r, _) in future_to_task.items():
                            if not f.done():
                                in_flight.append((now - submit_times[f], p, r))

                        if in_flight:
                            in_flight.sort(reverse=True, key=lambda x: x[0])
                            n_show = min(3, len(in_flight))
                            tqdm.write(f"  >> {len(in_flight)} tasks in-flight ({dataset_name}). Longest running:")
                            for rank, (t, p, r) in enumerate(in_flight[:n_show], 1):
                                tqdm.write(
                                    f"     {rank}. {t:.0f}s elapsed | run={r} | "
                                    f"{self._format_params_short(p)}"
                                )

                        if completed % 50 == 0:
                            self.save_intermediate_results(results, f"{dataset_name}_intermediate")

                    except Exception as e:
                        tqdm.write(f"  ERROR for task {task_idx} after {elapsed:.1f}s: {e}")
                        error_result = {
                            'run_id': run_id,
                            'num_particles': params.get('num_particles', -1),
                            'cluster_strategy': params.get('cluster_strategy', 'error'),
                            'positive_cluster_threshold': params.get('positive_cluster_threshold', -1),
                            'movement_strategy': params.get('movement_strategy', 'error'),
                            'initialization_strategy': params.get('initialization_strategy', 'error'),
                            'avg_node_pot_threshold': params.get('avg_node_pot_threshold', -1),
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
                        pbar.update(1)

        return results
    
    def _run_single_experiment_worker(self, 
                                     graph: nx.Graph, 
                                     num_neg: int,
                                     params: Dict[str, Any], 
                                     run_id: int,
                                     task_idx: int) -> Dict[str, Any]:
        """
        Worker function for parallel execution of single experiments.
        This function needs to be a standalone method for multiprocessing.
        
        Args:
            graph: NetworkX graph to use
            num_neg: Number of reliable negatives to select
            params: Parameter dictionary
            run_id: ID of this run
            task_idx: Index of this task
            
        Returns:
            Dictionary with experiment results
        """
        try:
            # Check if graph has true_label attribute for evaluation
            # Use safe node access since graph nodes may have gaps after processing
            node_attributes = dict(graph.nodes(data=True))
            if not any('true_label' in node_attributes[node] for node in graph.nodes()):
                print(f"Warning: Graph does not have 'true_label' attribute. Evaluation may not be accurate.")
            
            # Extract parameters
            pcm_params = {
                'num_particles': params['num_particles'],
                'movement_strategy': params['movement_strategy'],
                'initialization_strategy': params['initialization_strategy'],
                'average_node_potential_threshold': params['avg_node_pot_threshold']
            }
            
            rns_params = {
                'cluster_strategy': params['cluster_strategy'],
                'positive_cluster_threshold': params['positive_cluster_threshold']
            }
            
            # Initialize and train model
            model = PULearningPC(
                graph=graph,
                num_neg=num_neg,  # Use the parameter passed from command line
                pcm_params=pcm_params,
                rns_params=rns_params
            )
            
            # Train the model
            model.train()
            
            # Select reliable negatives
            reliable_negatives = model.select_reliable_negatives()
            
            # Check if we got enough reliable negatives
            if not reliable_negatives:
                print(f"Warning: No reliable negatives found for run {run_id}. Marking as error.")
                result = {
                    'run_id': run_id,
                    'num_particles': params['num_particles'],
                    'cluster_strategy': params['cluster_strategy'],
                    'positive_cluster_threshold': params['positive_cluster_threshold'],
                    'movement_strategy': params['movement_strategy'],
                    'initialization_strategy': params['initialization_strategy'],
                    'avg_node_pot_threshold': params['avg_node_pot_threshold'],
                    'step1_f1_score': 0.0,
                    'step1_precision': 0.0,
                    'step1_recall': 0.0,
                    'step1_num_reliable_negatives': 0,
                    'step2_f1': 0.0,
                    'step2_precision': 0.0,
                    'step2_recall': 0.0,
                    'step2_accuracy': 0.0,
                    'coverage': model.pcm.get_graph_coverage()[1],
                    'fallback_rule_used': model.labeled_graph.graph.get('fallback_rule_used', False),
                    'avg_threshold_criteria': model.labeled_graph.graph.get('avg_threshold_criteria', False),
                    'final_iteration_count': model.labeled_graph.graph.get('final_iteration_count', 0),
                    'graph_nodes': graph.number_of_nodes(),
                    'graph_edges': graph.number_of_edges(),
                    'status': 'error: insufficient reliable negatives'
                }
                return result

            # Step 1 metrics: evaluate reliable negative purity
            metrics = evaluate_reliable_negative_metrics(
                graph, reliable_negatives
            )

            # Step 2: full graph classification via harmonic label propagation
            predictions = model.classify(reliable_negatives)
            step2_metrics = evaluate_phase2(graph, predictions)

            # Get final coverage from the model
            final_coverage = model.pcm.get_graph_coverage()[1]

            fallback_rule_used = model.labeled_graph.graph.get('fallback_rule_used', False)
            avg_threshold_criteria = model.labeled_graph.graph.get('avg_threshold_criteria', False)
            final_iteration_count = model.labeled_graph.graph.get('final_iteration_count', 0)

            # Compile results (Step 1 + Step 2 metrics)
            result = {
                'run_id': run_id,
                'num_particles': params['num_particles'],
                'cluster_strategy': params['cluster_strategy'],
                'positive_cluster_threshold': params['positive_cluster_threshold'],
                'movement_strategy': params['movement_strategy'],
                'initialization_strategy': params['initialization_strategy'],
                'avg_node_pot_threshold': params['avg_node_pot_threshold'],
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
            }

            return result

        except Exception as e:
            result = {
                'run_id': run_id,
                'num_particles': params.get('num_particles', -1),
                'cluster_strategy': params.get('cluster_strategy', 'error'),
                'positive_cluster_threshold': params.get('positive_cluster_threshold', -1),
                'movement_strategy': params.get('movement_strategy', 'error'),
                'initialization_strategy': params.get('initialization_strategy', 'error'),
                'avg_node_pot_threshold': params.get('avg_node_pot_threshold', -1),
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
            return result

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

        summary = df.groupby([
            'num_particles', 'cluster_strategy',
            'positive_cluster_threshold', 'movement_strategy', 'initialization_strategy',
            'avg_node_pot_threshold'
        ]).agg(agg_dict).round(4)
        
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