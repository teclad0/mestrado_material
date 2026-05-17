# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master's research project on **Positive-Unlabeled (PU) Learning on graphs**. The goal is to identify reliable negative examples from unlabeled nodes in a graph where only a subset of positive nodes are labeled (SCAR assumption — Selected Completely at Random). The project implements a novel **Particle Competition-based algorithm** called **CP-APNR** (Competição de Partículas para Aprendizado Positivo e Não Rotulado), implemented as `PULearningPC` in code, and benchmarks it against existing methods.

- **Conferência alvo**: ENIAC/SBC (template em `template-latex/`)
- **Base**: Dissertação de mestrado (UFSCar, orientador: Alan Demétrius Baria Valejo)

## Environment Setup

```bash
# Uses a local virtualenv (Python 3.10)
source env/bin/activate
pip install -r requirements.txt
```

Key dependencies: PyTorch, PyTorch Geometric, networkit, scikit-learn, networkx, matplotlib, plotly, tqdm.

## Running Experiments

```bash
# Generate a dataset (saved as pickle of NetworkX graph)
python dataset_system.py --dataset cora --n-samples 1000

# Run parametric experiments for PULearningPC (grid search over hyperparameters)
python run_experiments.py --dataset cora --n-runs 5
python run_experiments.py --dataset cora --quick-test  # faster subset
python run_experiments.py --dataset cora citeseer --output-dir my_results  # multiple datasets

# Run all baseline models on a single dataset
python run_single_dataset_experiments.py <dataset_name> [n_samples] [percent_positive] [n_runs]
# Example: python run_single_dataset_experiments.py cora 1000 0.1 10
```

Notebooks: `running_models.ipynb` runs all models on datasets; `datasets.ipynb` analyzes dataset properties; `resultados.ipynb` analyzes experiment results.

## Architecture

### Algorithm Implementations (the core of the project)

**`model.py`** — The author's novel algorithm, structured as a 2-step pipeline:
1. `ParticleCompetitionModel` — Particles traverse the graph competing for node ownership, forming clusters. Uses `core.py::Particle` for particle state and `core.py::OrderedSet` for tracking visited nodes.
2. `ReliableNegativeSelection` — Labels clusters as positive/negative, then ranks negative-cluster nodes by dissimilarity (shortest path distance) to positive clusters.
3. `PULearningPC` — Wrapper class that orchestrates both steps. Takes a **NetworkX graph** directly.

**`models.py`** — Baseline algorithms, all following the same interface (`__init__`, `train()`, `negative_inference(num_neg)`):
- `RCSVM` — Cosine-similarity based representant vectors
- `CCRNE` — Clustering-based with radius threshold
- `PU_LP` — Label propagation on adjacency matrix (uses `(I - αA)^{-1}`)
- `LP_PUL` — BFS-based distance from positives (requires networkit)
- `MCLS` — KMeans clustering + cluster labeling

All baseline models accept either a NetworkX graph or a PyG Data object — they detect the type in `__init__` and convert internally.

**`models_experiment.py`** — Orchestrates running all baselines + PULearningPC together via `run_multiple_experiments()` and `save_results_to_csv()`. Used by `run_single_dataset_experiments.py`.

### Data Input Conventions

All algorithms accept one of two input formats:
- **NetworkX graph** with node attributes: `features` (numpy array), `true_label` (0/1), `observed_label` (0/1 where 1 = labeled positive)
- **PyG Data object** with `x`, `edge_index`, `P` (positive indices), `U` (unlabeled indices)

`dataset_system.py::DatasetManager` handles conversion between these formats (`get_data_for_mcls()`, `get_data_for_lp_pul()`).

### Dataset Generation

**`generate_dataset.py`** — Core dataset generation functions `load_<name>_scar()` for each dataset (cora, citeseer, twitch, mnist). Each:
1. Loads raw data (Planetoid for citation networks, OpenML for tabular)
2. Builds a k-NN or original-edge graph
3. Applies SCAR labeling via `apply_scar_labeling()`
4. Ensures connectivity via MST or largest component extraction

**Two dataset management systems exist** (both ultimately call `generate_dataset.py`):
- `dataset_system.py::DatasetManager` — Saves datasets as **pickle files**. This is the primary system used for experiments. Pickle files go to `datasets/` with naming convention `<name>_full_<pct>pct.pkl` (e.g., `cora_full_25pct.pkl`).
- `dataset_generator.py` / `dataset_loader.py` — Alternative JSON-based system for consistent cross-model experiments.

### Experiment Infrastructure

- **`experiment_config.py`** — Central configuration: `EXPERIMENT_CONFIG` (n_runs, output_dir, seed), `DATASET_CONFIG` (per-dataset params like k, positive_class_label, percent_positive), and parameter grid for PULearningPC.
- **`parametric_experiments.py`** — `PULearningExperimentRunner`: grid search over PULearningPC hyperparameters with parallel execution via `ProcessPoolExecutor`.
- **`run_experiments.py`** — CLI wrapper for parametric experiments.
- **`aux.py`** — Helper functions: `prepare_pu_graph_data()` for SCAR labeling and `dict_datasets_params_pulpc()` for per-dataset default PULearningPC parameters (cluster strategy, thresholds, num_neg).
- **`analyze_results.py`** — `ExperimentResultsAnalyzer`: post-experiment analysis and chart generation from CSV results.

Results are saved to `experiment_results/` as CSV files (both `_detailed_results.csv` and `_summary_results.csv`).

## Key Conventions

- The project language mixes Portuguese and English (variable names, comments, docs)
- Evaluation metrics: F1-score, precision, recall (from sklearn) computed on reliable negatives vs true labels
- `num_neg` parameter controls how many reliable negatives each algorithm returns — critical for fair benchmarking
- Datasets are persisted as pickle files to ensure all algorithms are evaluated on identical data splits
- Random seed is typically 42; set in `experiment_config.py` and propagated through `DatasetManager`
- **Paper fidelity**: When implementing algorithms from papers, ALWAYS follow the paper's described method faithfully. Never substitute a simpler heuristic (e.g., median thresholding) in place of the paper's actual algorithm (e.g., label propagation). If the paper's method is too complex or unclear, explicitly tell the user that you are deviating and explain why — do not silently take an easier route.
- **Regression testing on code changes**: When modifying a model's implementation (e.g., optimizing, refactoring, or fixing an algorithm), ALWAYS compare outputs before and after the change on a known dataset to verify correctness is preserved. Run the model on at least one dataset (e.g., Cora) before and after, and confirm that results match or improve. This catches silent regressions where "working code" produces wrong results.

## Article Writing Wiki

A knowledge base in `wiki/` supports writing the ENIAC/SBC article about CP-APNR.

### Folder Structure

```
raw/              -- documentos fonte (NUNCA modificar)
wiki/             -- páginas da wiki mantidas pelo Claude
wiki/index.md     -- índice geral
wiki/log.md       -- registro de operações
template-latex/   -- template LaTeX do artigo SBC
```

### Fontes em raw/

| Arquivo | Conteúdo | Papel no artigo |
|---------|----------|-----------------|
| `Dissertacao_Mestrado_Qualificacao_Hemilyn_Stephanye (1).pdf` | Qualificação de mestrado | **Fonte principal** — método, experimentos, resultados |
| `Particle competition for complex network community detection.pdf` | Algoritmo PC original | Fundamentação do método proposto |
| `Clustering-based method for positive and unlabeled text categoriza.pdf` | Algoritmo C-CRNE | Trabalho relacionado / baseline |
| `l. A graph-based approach for positive and unlabeled learning.pdf` | Algoritmo LP-PUL | Trabalho relacionado / baseline |
| `learning to classify texts using positive and unlabeled data..pdf` | Algoritmo RC-SVM | Trabalho relacionado / baseline |
| `MCLS.pdf` | Algoritmo MCLS | Trabalho relacionado / baseline |
| `Pu-lp: A novel approach for positive and unlabeled learning by label.pdf` | Algoritmo PU-LP | Trabalho relacionado / baseline |
| `DISSERTAÇÃO RI FINAL GUILHERME MESSIAS_colega.pdf` | Dissertação colega | Referência auxiliar |

### Wiki Organization

The wiki is organized around **article sections**:

- **Conceitos centrais**: `pu-learning.md`, `reliable-negatives.md`, `particle-competition.md`, `cp-apnr.md`
- **Algoritmos baseline**: `c-crne.md`, `lp-pul.md`, `rc-svm.md`, `mcls.md`, `pu-lp.md`
- **Experimentos**: `datasets.md`, `metricas.md`, `resultados.md`
- **Rascunhos do artigo**: `draft-*.md` files (introduction, related work, method, experiments, conclusion)

### Wiki Page Format

```markdown
# Título

**Resumo**: Uma ou duas frases.
**Fontes**: Arquivos de raw/ que alimentam esta página.
**Última atualização**: Data.
**Relevância para o artigo**: Em qual seção do artigo este conteúdo aparece.

---

Conteúdo principal com [[wiki-links]].

## Páginas relacionadas
- [[conceito-1]]
```

### Draft Format (draft-*.md)

```markdown
# Seção: Nome da Seção

**Status**: rascunho | em revisão | aprovado
**Versão**: número
**Última atualização**: Data.

---

Texto do artigo com referência à fonte: (ver: [[página-wiki]])

## Notas para revisão
- Pontos em aberto ou dúvidas
```

### Writing Workflow

1. **Ingerir fonte**: Ler PDF → criar/atualizar páginas conceituais no wiki
2. **Sintetizar**: Cruzar informações entre páginas para identificar gaps
3. **Rascunhar**: Escrever seções do artigo em `draft-*.md` usando o wiki como base
4. **Revisar**: Iterar rascunhos com feedback do usuário
5. **Finalizar**: Transferir texto aprovado para o LaTeX em `template-latex/`

### Wiki Rules

- Nunca modificar arquivos em `raw/`
- Sempre atualizar `wiki/index.md` e `wiki/log.md` após mudanças
- Nomes de arquivo em lowercase com hífens
- Toda claim factual deve referenciar o arquivo fonte
- Rascunhos do artigo devem seguir o estilo acadêmico SBC
- Quando incerto, perguntar ao usuário
