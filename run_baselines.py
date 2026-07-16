#!/usr/bin/env python3
"""
Run all baseline PU Learning models (Phase 1 + Phase 2) on configured datasets.

Usage:
    python run_baselines.py                          # all models, all datasets, 5 runs
    python run_baselines.py --dataset cora           # single dataset
    python run_baselines.py --model RCSVM            # single model
    python run_baselines.py --percent 0.25           # single percent_positive
    python run_baselines.py --n-jobs 8               # limit parallelism
"""

import argparse
import os
import time
import warnings
import multiprocessing
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')

from models import RCSVM, CCRNE, PU_LP, LP_PUL, MCLS
from dataset_system import DatasetManager
from models_experiment import evaluate_f1_score, evaluate_precision_score, evaluate_phase2, classify_with_svm
from experiment_config import BASELINE_CONFIG


def run_phase1(model, graph, num_neg, needs_mapping=False, node_list=None):
    model.train()
    rn = model.negative_inference(num_neg)
    rn_list = rn.tolist() if isinstance(rn, torch.Tensor) else list(rn)

    if needs_mapping and node_list is not None:
        rn_eval = [node_list[i] for i in rn_list if i < len(node_list)]
    else:
        rn_eval = rn_list

    f1 = evaluate_f1_score(graph, rn_eval, target_negatives=num_neg)
    precision = evaluate_precision_score(graph, rn_eval, target_negatives=num_neg)
    return rn_list, rn_eval, f1, precision


def run_phase2(rn_eval, graph):
    predictions = classify_with_svm(graph, rn_eval)
    return evaluate_phase2(graph, predictions)


def instantiate_model(model_name, graph, manager, params):
    if model_name == 'RCSVM':
        return RCSVM(graph, **params), False, None
    elif model_name == 'CCRNE':
        return CCRNE(graph, **params), False, None
    elif model_name == 'PU_LP':
        return PU_LP(graph, **params), False, None
    elif model_name == 'MCLS':
        node_list = sorted(list(graph.nodes()))
        mcls_data = manager.get_data_for_mcls(graph)
        return MCLS(mcls_data, **params), True, node_list
    elif model_name == 'LP_PUL':
        node_list = sorted(list(graph.nodes()))
        lp_pul_data = manager.get_data_for_lp_pul(graph)
        return LP_PUL(lp_pul_data), True, node_list
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_single_experiment(model_name, graph, manager, num_neg, params):
    model, needs_mapping, node_list = instantiate_model(model_name, graph, manager, params)

    t0 = time.time()
    rn_list, rn_eval, f1_p1, prec_p1 = run_phase1(model, graph, num_neg, needs_mapping, node_list)
    t_p1 = time.time() - t0

    t0 = time.time()
    phase2 = run_phase2(rn_eval, graph)
    t_p2 = time.time() - t0

    return {
        'phase1_f1': f1_p1,
        'phase1_precision': prec_p1,
        'phase1_num_rn': len(rn_list),
        'phase1_time_s': t_p1,
        'phase2_f1': phase2['f1'],
        'phase2_precision': phase2['precision'],
        'phase2_recall': phase2['recall'],
        'phase2_accuracy': phase2['accuracy'],
        'phase2_time_s': t_p2,
    }


def _worker(task):
    """Top-level worker — must be module-level to be picklable by ProcessPoolExecutor."""
    model_name, dataset, pct, run = task
    warnings.filterwarnings('ignore')

    pct_str = f"{int(pct * 100)}pct"
    path = f"datasets/{dataset}_full_{pct_str}_run_{run}.pkl"

    if not os.path.exists(path):
        return None

    try:
        manager = DatasetManager()
        num_neg = BASELINE_CONFIG['num_neg'][dataset]
        params = BASELINE_CONFIG['models'][model_name]

        graph = manager.load_graph(path)
        res = run_single_experiment(model_name, graph, manager, num_neg, params)
        res.update({'model': model_name, 'dataset': dataset, 'percent_positive': pct, 'run': run})
        return res
    except Exception as e:
        return {
            'model': model_name, 'dataset': dataset,
            'percent_positive': pct, 'run': run,
            'error': str(e),
        }


def main():
    parser = argparse.ArgumentParser(description='Run baseline PU Learning models')
    parser.add_argument('--dataset', type=str, default=None, help='Single dataset to run')
    parser.add_argument('--model', type=str, default=None, help='Single model to run')
    parser.add_argument('--percent', type=float, default=None, help='Single percent_positive to run')
    parser.add_argument('--n-runs', type=int, default=5, help='Number of runs (default: 5)')
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='Parallel workers (default: min(cpu_count, 16))')
    parser.add_argument('--output', type=str, default='baseline_results.csv', help='Output CSV path')
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else BASELINE_CONFIG['datasets']
    percents = [args.percent] if args.percent else BASELINE_CONFIG['percent_positive']
    models = [args.model] if args.model else list(BASELINE_CONFIG['models'].keys())
    n_jobs = args.n_jobs or min(multiprocessing.cpu_count(), 16)

    # Build the full task list — every combination is independent
    tasks = [
        (model_name, dataset, pct, run)
        for model_name in models
        for dataset in datasets
        for pct in percents
        for run in range(args.n_runs)
    ]

    print(f"Tasks: {len(tasks)} | Workers: {n_jobs}")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Percent positives: {percents}")
    print(f"Runs per combo: {args.n_runs}")
    print()

    results = []
    completed = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_worker, task): task for task in tasks}

        for future in as_completed(futures):
            task = futures[future]
            model_name, dataset, pct, run = task
            completed += 1

            try:
                res = future.result()
            except Exception as e:
                res = {'model': model_name, 'dataset': dataset,
                       'percent_positive': pct, 'run': run, 'error': str(e)}

            if res is None:
                pct_str = f"{int(pct * 100)}pct"
                print(f"  [{completed}/{len(tasks)}] SKIP {model_name} {dataset}/{pct_str} run={run} — file not found")
            elif 'error' in res:
                pct_str = f"{int(pct * 100)}pct"
                print(f"  [{completed}/{len(tasks)}] ERROR {model_name} {dataset}/{pct_str} run={run}: {res['error']}")
                results.append(res)
            else:
                elapsed = time.time() - t_start
                pct_str = f"{int(pct * 100)}pct"
                print(f"  [{completed}/{len(tasks)}] {model_name:6s} {dataset:8s}/{pct_str} run={run}: "
                      f"P1={res['phase1_f1']:.3f} P2={res['phase2_f1']:.3f} "
                      f"({res['phase1_time_s'] + res['phase2_time_s']:.1f}s) | total={elapsed:.0f}s")
                results.append(res)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        total_time = time.time() - t_start
        print(f"\nDone in {total_time:.0f}s. Results saved to {args.output}")

        successful = df[~df.get('error', pd.Series(dtype=str)).notna()] if 'error' in df.columns else df
        print(f"\n{'=' * 60}")
        print("SUMMARY (mean ± std across runs)")
        print(f"{'=' * 60}")
        if 'phase1_f1' in df.columns:
            summary = (df.groupby(['model', 'dataset', 'percent_positive'])['phase1_f1']
                       .agg(['mean', 'std']).reset_index())
            print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
