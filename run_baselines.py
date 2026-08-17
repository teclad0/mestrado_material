#!/usr/bin/env python3
"""
Run all baseline PU Learning models plus PULearningPC (Phase 1 + Phase 2) on
configured datasets, all through one script so Table 4 / Table 5 data lands in
a single CSV.

Usage:
    python run_baselines.py                          # all models, all datasets, 5 runs
    python run_baselines.py --dataset cora           # single dataset
    python run_baselines.py --model RCSVM            # single model
    python run_baselines.py --model PULearningPC     # just the proposed method
    python run_baselines.py --percent 0.25           # single percent_positive
    python run_baselines.py --n-jobs 8               # limit parallelism
"""

import argparse
import contextlib
import io
import os
import random
import time
import warnings
import multiprocessing
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

warnings.filterwarnings('ignore')

from models import RCSVM, CCRNE, PU_LP, LP_PUL, MCLS
from model import PULearningPC
from dataset_system import DatasetManager
from models_experiment import (
    evaluate_f1_score, evaluate_precision_score, evaluate_phase2,
    classify_with_svm, classify_with_llgc,
)
from experiment_config import BASELINE_CONFIG, DATASET_GEN_PARAMS
from aux import dict_datasets_params_pulpc

# Graph-based Phase-1 methods use LLGC (Zhou et al., 2003) for Phase 2 instead of
# the shared SVM — they already operate on the graph structure, so propagating
# P+RN labels over that same graph is a more natural fit than a feature-space SVM.
GRAPH_BASED_MODELS = {'PU_LP', 'LP_PUL', 'PULearningPC'}

# Models with a stochastic Phase 1 (unseeded internal randomness) — these need
# repeated runs (against the SAME fixed dataset/labeling) to average out their own
# variance. Everything else is deterministic given a fixed graph + labeling, so
# repeating them against that same input just reproduces the same result.
#
# NOTE: this measures algorithm-internal variance, not the positive-sample-selection
# variance the paper's Tables 4/5 report (Sec 4.1: "3 execuções com diferentes
# amostras de positivos"). Deterministic models will therefore show std=0 here,
# diverging from the paper's reported std for those rows — a deliberate tradeoff
# to cut compute, chosen over regenerating a fresh SCAR sample per run.
STOCHASTIC_MODELS = {'MCLS', 'PULearningPC'}


def n_runs_for(model_name: str, requested_n_runs: int) -> int:
    return requested_n_runs if model_name in STOCHASTIC_MODELS else 1


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


def run_phase2(rn_eval, graph, model_name):
    """Phase-2 classifier, chosen per model family.

    Per the paper (Sec 4, "Avaliação de Desempenho"): baselines use the shared
    RBF-kernel SVM trained on P + RN. Graph-based Phase-1 methods (see
    GRAPH_BASED_MODELS) instead use LLGC (Zhou et al., 2003), propagating P+RN
    labels over the same graph they already operate on in Phase 1.
    """
    if model_name in GRAPH_BASED_MODELS:
        predictions = classify_with_llgc(graph, rn_eval)
    else:
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


def run_single_experiment_pulpc(graph, num_neg, dataset_name):
    """PULearningPC has a different Phase-1 interface (train() + select_reliable_negatives()
    instead of negative_inference()), so it can't go through instantiate_model/run_phase1.

    pcm_params match the paper's final-comparison configuration (Sec 4.2, end of
    parametric analysis): n_p=100, random init, uniform exploration, theta_avg=0.8.
    rns_params come from the same per-dataset defaults (aux.dict_datasets_params_pulpc)
    used everywhere else in this codebase.
    """
    dataset_params = dict_datasets_params_pulpc().get(dataset_name, {})
    rns_params = dataset_params.get('rns_params', {
        'cluster_strategy': 'percentage',
        'positive_cluster_threshold': 0.1
    })
    pcm_params = {
        'num_particles': 100,
        'movement_strategy': 'uniform',
        'initialization_strategy': 'random',
        'average_node_potential_threshold': 0.8
    }

    model = PULearningPC(graph, num_neg=num_neg, pcm_params=pcm_params, rns_params=rns_params)

    t0 = time.time()
    model.train()
    reliable_negatives = model.select_reliable_negatives()
    f1_p1 = evaluate_f1_score(graph, reliable_negatives, target_negatives=num_neg)
    prec_p1 = evaluate_precision_score(graph, reliable_negatives, target_negatives=num_neg)
    t_p1 = time.time() - t0

    t0 = time.time()
    phase2 = run_phase2(reliable_negatives, graph, 'PULearningPC')
    t_p2 = time.time() - t0

    return {
        'phase1_f1': f1_p1,
        'phase1_precision': prec_p1,
        'phase1_num_rn': len(reliable_negatives),
        'phase1_time_s': t_p1,
        'phase2_f1': phase2['f1'],
        'phase2_precision': phase2['precision'],
        'phase2_recall': phase2['recall'],
        'phase2_accuracy': phase2['accuracy'],
        'phase2_time_s': t_p2,
    }


def run_single_experiment(model_name, graph, manager, num_neg, params, dataset_name):
    if model_name == 'PULearningPC':
        return run_single_experiment_pulpc(graph, num_neg, dataset_name)

    model, needs_mapping, node_list = instantiate_model(model_name, graph, manager, params)

    t0 = time.time()
    rn_list, rn_eval, f1_p1, prec_p1 = run_phase1(model, graph, num_neg, needs_mapping, node_list)
    t_p1 = time.time() - t0

    t0 = time.time()
    phase2 = run_phase2(rn_eval, graph, model_name)
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
    """Top-level worker — must be module-level to be picklable by ProcessPoolExecutor.

    The dataset (graph structure + SCAR labeling) is fixed per (dataset, percent_positive)
    and reused across all runs — only each algorithm's own internal randomness (e.g.
    MCLS's unseeded KMeans, PULearningPC's particle movement) varies between runs.
    Seeding per run keeps that variation reproducible across invocations.
    """
    model_name, dataset, pct, run, num_neg_mode = task
    warnings.filterwarnings('ignore')

    pct_str = f"{int(pct * 100)}pct"
    path = f"datasets/{dataset}_full_{pct_str}.pkl"

    if not os.path.exists(path):
        return None

    try:
        # DatasetManager.__init__ unconditionally reseeds random/np.random/torch to
        # a fixed seed (42) — construct it and load the graph BEFORE applying our
        # per-run seed below, otherwise that fixed reseed clobbers the per-run
        # variation and every "stochastic" run collapses to the same result.
        manager = DatasetManager()
        graph = manager.load_graph(path)

        random.seed(42 + run)
        np.random.seed(42 + run)
        torch.manual_seed(42 + run)

        if num_neg_mode == 'positives':
            # Match num_neg to the actual number of observed positives for this
            # (dataset, percent_positive) combo, instead of the fixed per-dataset
            # value in BASELINE_CONFIG — e.g. citeseer at 1% has only 5 labeled
            # positives, so num_neg=5 there instead of the fixed 200.
            num_neg = sum(1 for _, attrs in graph.nodes(data=True) if attrs.get('observed_label') == 1)
        else:
            num_neg = BASELINE_CONFIG['num_neg'][dataset]
        params = BASELINE_CONFIG['models'].get(model_name, {})
        if model_name == 'MCLS':
            k_override = BASELINE_CONFIG.get('MCLS_k_per_dataset', {}).get(dataset)
            if k_override is not None:
                params = {**params, 'k': k_override}

        # model.py / models.py print verbose internal progress that interleaves
        # unreadably across parallel workers — suppress it here.
        with contextlib.redirect_stdout(io.StringIO()):
            res = run_single_experiment(model_name, graph, manager, num_neg, params, dataset)
        res.update({'model': model_name, 'dataset': dataset, 'percent_positive': pct, 'run': run})
        return res
    except Exception as e:
        return {
            'model': model_name, 'dataset': dataset,
            'percent_positive': pct, 'run': run,
            'error': str(e),
        }


def main():
    parser = argparse.ArgumentParser(description='Run baseline PU Learning models + PULearningPC')
    parser.add_argument('--dataset', type=str, default=None, help='Single dataset to run')
    parser.add_argument('--model', type=str, default=None, help='Single model to run')
    parser.add_argument('--percent', type=float, default=None, help='Single percent_positive to run')
    parser.add_argument('--n-runs', type=int, default=5, help='Number of runs (default: 5)')
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='Parallel workers (default: min(cpu_count, 16))')
    parser.add_argument('--output', type=str, default='baseline_results.csv', help='Output CSV path')
    parser.add_argument('--num-neg-mode', type=str, default='fixed', choices=['fixed', 'positives'],
                        help="'fixed' (default): num_neg from BASELINE_CONFIG per dataset. "
                             "'positives': num_neg = number of observed positives for that "
                             "(dataset, percent_positive) combo.")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else BASELINE_CONFIG['datasets']
    percents = [args.percent] if args.percent else BASELINE_CONFIG['percent_positive']
    all_models = list(BASELINE_CONFIG['models'].keys()) + ['PULearningPC']
    models = [args.model] if args.model else all_models
    n_jobs = args.n_jobs or min(multiprocessing.cpu_count(), 16)

    # Generate any missing (dataset, percent_positive) files up front, in the main
    # process — one file is reused across all models/runs for that combo. Doing this
    # here (not inside _worker) avoids multiple pool workers racing to generate and
    # write the same file at once.
    manager = DatasetManager()
    for dataset in datasets:
        for pct in percents:
            pct_str = f"{int(pct * 100)}pct"
            path = f"datasets/{dataset}_full_{pct_str}.pkl"
            if not os.path.exists(path):
                print(f"Generating missing dataset: {path}")
                manager.generate_and_save_dataset(dataset, DATASET_GEN_PARAMS.get(dataset, {}), None, pct)

    # Build the full task list — every combination is independent.
    # Deterministic models only need 1 run per (dataset, percent_positive); repeating
    # them against the same fixed input would just reproduce the same result.
    # Stochastic models (see STOCHASTIC_MODELS) get the full --n-runs.
    tasks = [
        (model_name, dataset, pct, run, args.num_neg_mode)
        for model_name in models
        for dataset in datasets
        for pct in percents
        for run in range(n_runs_for(model_name, args.n_runs))
    ]

    print(f"Tasks: {len(tasks)} | Workers: {n_jobs}")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Percent positives: {percents}")
    print(f"Runs per combo: {args.n_runs} (deterministic models run once regardless)")
    print()

    results = []
    completed = 0
    t_start = time.time()

    # Use 'spawn' instead of the default 'fork': by this point torch has already
    # been used in the main process (e.g. DatasetManager's torch.manual_seed call),
    # which can leave PyTorch's internal thread pool holding a lock at fork time —
    # forked workers then deadlock forever waiting on it. spawn starts clean
    # interpreters instead, avoiding the inherited-lock issue entirely.
    mp_context = multiprocessing.get_context('spawn')

    # A worker can be killed abruptly (e.g. OOM) independently of any task's own
    # correctness. When that happens every other future already queued on that
    # same pool — including ones that hadn't even started — raises
    # BrokenProcessPool too, not just the task that actually died. Treating that
    # as a normal per-task error would wrongly mark hundreds of never-run tasks
    # as failed. Instead, on a broken pool we start a fresh one and retry
    # whatever's left, up to a small retry cap per task.
    #
    # Submitting all tasks to a single long-lived executor makes the blast radius
    # of one bad task the *entire remaining task list* (everything still queued
    # inside that executor, not just what's concurrently running). Batching caps
    # that blast radius at BATCH_SIZE: each batch gets its own executor, so a
    # break only poisons the handful of tasks in that batch.
    retry_counts = {task: 0 for task in tasks}
    max_retries = 3
    pool_restarts = 0

    BATCH_SIZE = 8 * n_jobs
    batches = [tasks[i:i + BATCH_SIZE] for i in range(0, len(tasks), BATCH_SIZE)]

    for batch in batches:
        pending = list(batch)

        while pending:
            done_this_round = set()
            broke = False

            try:
                with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp_context) as executor:
                    futures = {executor.submit(_worker, task): task for task in pending}

                    for future in as_completed(futures):
                        task = futures[future]
                        model_name, dataset, pct, run, _ = task

                        try:
                            res = future.result()
                        except BrokenProcessPool:
                            raise
                        except Exception as e:
                            res = {'model': model_name, 'dataset': dataset,
                                   'percent_positive': pct, 'run': run, 'error': str(e)}

                        done_this_round.add(task)
                        completed += 1
                        elapsed = time.time() - t_start
                        pct_str = f"{int(pct * 100)}pct"

                        if res is None:
                            print(f"  [{completed}/{len(tasks)}] SKIP {model_name} {dataset}/{pct_str} run={run} — file not found")
                        elif 'error' in res:
                            print(f"  [{completed}/{len(tasks)}] ERROR {model_name} {dataset}/{pct_str} run={run}: {res['error']}")
                            results.append(res)
                        else:
                            print(f"  [{completed}/{len(tasks)}] {model_name:14s} {dataset:12s}/{pct_str} run={run}: "
                                  f"P1={res['phase1_f1']:.3f} P2={res['phase2_f1']:.3f} "
                                  f"({res['phase1_time_s'] + res['phase2_time_s']:.1f}s) | total={elapsed:.0f}s")
                            results.append(res)

            except BrokenProcessPool:
                broke = True

            pending = [t for t in pending if t not in done_this_round]

            if broke and pending:
                pool_restarts += 1
                still_pending = []
                for t in pending:
                    retry_counts[t] += 1
                    if retry_counts[t] <= max_retries:
                        still_pending.append(t)
                    else:
                        model_name, dataset, pct, run, _ = t
                        completed += 1
                        print(f"  [{completed}/{len(tasks)}] ERROR {model_name} {dataset}/{int(pct*100)}pct run={run}: "
                              f"exceeded {max_retries} retries after BrokenProcessPool")
                        results.append({'model': model_name, 'dataset': dataset,
                                         'percent_positive': pct, 'run': run,
                                         'error': f'exceeded {max_retries} retries after BrokenProcessPool'})
                print(f"  [pool restart #{pool_restarts}] retrying {len(still_pending)} task(s)")
                pending = still_pending

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
