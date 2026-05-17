#!/usr/bin/env python3
"""
Run all baseline PU Learning models (Phase 1 + Phase 2) on configured datasets.

Usage:
    python run_baselines.py                     # all models, all datasets
    python run_baselines.py --dataset cora      # single dataset
    python run_baselines.py --model RCSVM       # single model
    python run_baselines.py --percent 0.25      # single percent_positive
"""

import argparse
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')

from models import RCSVM, CCRNE, PU_LP, LP_PUL, MCLS
from dataset_system import DatasetManager
from models_experiment import evaluate_f1_score, evaluate_precision_score, evaluate_phase2
from experiment_config import BASELINE_CONFIG, DATASET_CONFIG


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


def run_phase2(model, rn_list, graph, needs_mapping=False, node_list=None):
    predictions = model.classify(rn_list)
    if needs_mapping and node_list is not None:
        predictions = {node_list[k]: v for k, v in predictions.items() if k < len(node_list)}
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
    phase2 = run_phase2(model, rn_list, graph, needs_mapping, node_list)
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


def main():
    parser = argparse.ArgumentParser(description='Run baseline PU Learning models')
    parser.add_argument('--dataset', type=str, default=None, help='Single dataset to run')
    parser.add_argument('--model', type=str, default=None, help='Single model to run')
    parser.add_argument('--percent', type=float, default=None, help='Single percent_positive to run')
    parser.add_argument('--output', type=str, default='baseline_results.csv', help='Output CSV path')
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else BASELINE_CONFIG['datasets']
    percents = [args.percent] if args.percent else BASELINE_CONFIG['percent_positive']
    models = [args.model] if args.model else list(BASELINE_CONFIG['models'].keys())

    manager = DatasetManager()
    results = []

    for model_name in models:
        params = BASELINE_CONFIG['models'][model_name]
        print(f"\n{'=' * 60}")
        print(f"  {model_name}")
        print(f"{'=' * 60}")

        for dataset in datasets:
            num_neg = BASELINE_CONFIG['num_neg'][dataset]

            for pct in percents:
                pct_str = f"{int(pct * 100)}pct"
                path = f"datasets/{dataset}_full_{pct_str}.pkl"

                if not os.path.exists(path):
                    print(f"  SKIP {dataset}/{pct_str} — {path} not found")
                    continue

                graph = manager.load_graph(path)

                try:
                    res = run_single_experiment(model_name, graph, manager, num_neg, params)
                    res.update({
                        'model': model_name,
                        'dataset': dataset,
                        'percent_positive': pct,
                    })
                    results.append(res)

                    print(f"  {dataset}/{pct_str}: "
                          f"P1 F1={res['phase1_f1']:.4f} ({res['phase1_time_s']:.1f}s) | "
                          f"P2 F1={res['phase2_f1']:.4f} Acc={res['phase2_accuracy']:.4f} ({res['phase2_time_s']:.1f}s)")

                except Exception as e:
                    print(f"  ERROR {dataset}/{pct_str}: {e}")
                    results.append({
                        'model': model_name,
                        'dataset': dataset,
                        'percent_positive': pct,
                        'error': str(e),
                    })

    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")

        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        summary_cols = ['model', 'dataset', 'percent_positive', 'phase1_f1', 'phase2_f1', 'phase2_accuracy']
        available_cols = [c for c in summary_cols if c in df.columns]
        print(df[available_cols].to_string(index=False))


if __name__ == '__main__':
    main()
