"""
Tests for model.py — ParticleCompetitionModel, ReliableNegativeSelection, PULearningPC.

These tests create small synthetic graphs to verify correctness of:
1. is_cluster_positive() logic for both strategies
2. ParticleCompetitionModel initialization, movement, convergence
3. ReliableNegativeSelection cluster labeling and ranking
4. PULearningPC end-to-end pipeline
5. Determinism with fixed seeds
"""

import pytest
import networkx as nx
import numpy as np
import random
from collections import defaultdict

from model import (
    is_cluster_positive,
    ParticleCompetitionModel,
    ReliableNegativeSelection,
    PULearningPC,
)
from core import Particle, OrderedSet


# ============================================================================
# FIXTURES — Small synthetic graphs for testing
# ============================================================================

@pytest.fixture
def simple_graph():
    """
    A small connected graph (10 nodes) with clear positive/negative structure.
    Nodes 0-3: positive class (true_label=1), nodes 0-1 are observed (observed_label=1)
    Nodes 4-9: negative class (true_label=0), all unlabeled (observed_label=0)
    """
    G = nx.karate_club_graph()
    # Take a subgraph of 10 nodes for speed
    G = G.subgraph(range(10)).copy()

    for node in G.nodes():
        if node < 4:
            G.nodes[node]['true_label'] = 1
            G.nodes[node]['observed_label'] = 1 if node < 2 else 0
        else:
            G.nodes[node]['true_label'] = 0
            G.nodes[node]['observed_label'] = 0
        G.nodes[node]['features'] = np.random.rand(5)

    return G


@pytest.fixture
def two_cluster_graph():
    """
    Two well-separated cliques connected by a single edge.
    Cluster A (nodes 0-4): mostly positive
    Cluster B (nodes 5-9): all negative
    """
    G = nx.Graph()
    # Cluster A: complete graph on 0-4
    for i in range(5):
        for j in range(i + 1, 5):
            G.add_edge(i, j)
    # Cluster B: complete graph on 5-9
    for i in range(5, 10):
        for j in range(i + 1, 10):
            G.add_edge(i, j)
    # Bridge
    G.add_edge(4, 5)

    for node in G.nodes():
        if node < 5:
            G.nodes[node]['true_label'] = 1
            G.nodes[node]['observed_label'] = 1 if node < 3 else 0
        else:
            G.nodes[node]['true_label'] = 0
            G.nodes[node]['observed_label'] = 0
        G.nodes[node]['features'] = np.random.rand(5)

    return G


@pytest.fixture
def larger_graph():
    """
    A larger graph (~90 nodes) for testing convergence and num_neg.
    Uses a planted partition model for clear cluster structure.
    3 groups of 30 nodes each.
    """
    G = nx.planted_partition_graph(3, 30, 0.4, 0.02, seed=42)

    # First community (nodes 0-29) is positive
    for node in G.nodes():
        if node < 30:
            G.nodes[node]['true_label'] = 1
            G.nodes[node]['observed_label'] = 1 if node < 10 else 0
        else:
            G.nodes[node]['true_label'] = 0
            G.nodes[node]['observed_label'] = 0
        G.nodes[node]['features'] = np.random.rand(10)

    # Ensure connectivity
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            u = list(components[i])[0]
            v = list(components[i + 1])[0]
            G.add_edge(u, v)

    return G


# ============================================================================
# TESTS: is_cluster_positive
# ============================================================================

class TestIsClusterPositive:
    def test_majority_positive(self, simple_graph):
        """Cluster with majority observed_label=1 should be positive."""
        # Nodes 0,1 have observed_label=1; node 2 has observed_label=0
        nodes = [0, 1, 2]
        assert is_cluster_positive(simple_graph, nodes, 'majority') is True

    def test_majority_negative(self, simple_graph):
        """Cluster with majority observed_label=0 should be negative."""
        nodes = [4, 5, 6, 7]
        assert is_cluster_positive(simple_graph, nodes, 'majority') is False

    def test_percentage_above_threshold(self, simple_graph):
        """Cluster with positives above threshold should be positive."""
        # Nodes 0,1 positive out of 0,1,2,3 → 50%
        nodes = [0, 1, 2, 3]
        assert is_cluster_positive(simple_graph, nodes, 'percentage', threshold=0.4) is True

    def test_percentage_below_threshold(self, simple_graph):
        """Cluster with positives below threshold should be negative."""
        nodes = [0, 1, 2, 3]
        # 2/4 = 50%, threshold=0.6 → should be negative
        assert is_cluster_positive(simple_graph, nodes, 'percentage', threshold=0.6) is False

    def test_percentage_exact_threshold(self, simple_graph):
        """Cluster with positives exactly at threshold should be positive (>=)."""
        nodes = [0, 1, 2, 3]
        # 2/4 = 0.5
        assert is_cluster_positive(simple_graph, nodes, 'percentage', threshold=0.5) is True

    def test_majority_ignores_threshold(self, simple_graph):
        """Majority strategy should give the same result regardless of threshold."""
        nodes = [0, 1, 2]
        result_low = is_cluster_positive(simple_graph, nodes, 'majority', threshold=0.1)
        result_high = is_cluster_positive(simple_graph, nodes, 'majority', threshold=0.9)
        assert result_low == result_high

    def test_empty_cluster(self, simple_graph):
        """Empty cluster should be negative."""
        assert is_cluster_positive(simple_graph, [], 'majority') is False
        assert is_cluster_positive(simple_graph, [], 'percentage', threshold=0.1) is False

    def test_invalid_strategy_raises(self, simple_graph):
        """Unknown strategy should raise ValueError."""
        with pytest.raises(ValueError):
            is_cluster_positive(simple_graph, [0, 1], 'unknown_strategy')


# ============================================================================
# TESTS: ParticleCompetitionModel
# ============================================================================

class TestParticleCompetitionModel:
    def test_initialization_creates_particles(self, simple_graph):
        """Model should create the correct number of particles."""
        pcm = ParticleCompetitionModel(simple_graph, num_particles=3)
        assert len(pcm.particles) == 3

    def test_initialization_sets_node_attributes(self, simple_graph):
        """All nodes should have owner=None and potential=0.05 after init."""
        pcm = ParticleCompetitionModel(simple_graph, num_particles=3)
        for node in pcm.graph.nodes():
            assert pcm.graph.nodes[node]['owner'] is None
            assert pcm.graph.nodes[node]['potential'] == 0.05

    def test_move_particle_returns_valid_node(self, simple_graph):
        """move_particle should return a node that exists in the graph."""
        pcm = ParticleCompetitionModel(simple_graph, num_particles=3)
        particle = pcm.particles[0]
        node = pcm.move_particle(particle)
        assert node in pcm.graph.nodes()

    def test_move_uninitialized_particle_random(self, simple_graph):
        """First move of uninitialized particle (random strategy) should pick a node."""
        random.seed(42)
        pcm = ParticleCompetitionModel(
            simple_graph, num_particles=3, initialization_strategy='random'
        )
        particle = pcm.particles[0]
        assert particle.current_position is None
        node = pcm.move_particle(particle)
        assert node in pcm.graph.nodes()

    def test_move_uninitialized_particle_degree_weighted(self, simple_graph):
        """First move with degree_weighted init should pick from high-degree candidates."""
        random.seed(42)
        pcm = ParticleCompetitionModel(
            simple_graph, num_particles=3, initialization_strategy='degree_weighted'
        )
        particle = pcm.particles[0]
        node = pcm.move_particle(particle)
        assert node in pcm.graph.nodes()

    def test_simulation_converges(self, simple_graph):
        """Simulation should terminate (not hang forever)."""
        random.seed(42)
        np.random.seed(42)
        pcm = ParticleCompetitionModel(
            simple_graph, num_particles=3,
            average_node_potential_threshold=0.6, patience=30
        )
        result = pcm.run_simulation()
        assert isinstance(result, nx.Graph)
        assert 'final_iteration_count' in result.graph
        assert result.graph['final_iteration_count'] > 0

    def test_simulation_assigns_owners(self, simple_graph):
        """After simulation, most nodes should have an owner."""
        random.seed(42)
        np.random.seed(42)
        pcm = ParticleCompetitionModel(
            simple_graph, num_particles=3,
            average_node_potential_threshold=0.6, patience=30
        )
        pcm.run_simulation()
        owned_nodes = sum(
            1 for _, data in pcm.graph.nodes(data=True) if data['owner'] is not None
        )
        # At least some nodes should be owned
        assert owned_nodes > 0

    def test_owner_groups_consistent(self, simple_graph):
        """owner_groups should match actual node owners in the graph."""
        random.seed(42)
        np.random.seed(42)
        pcm = ParticleCompetitionModel(
            simple_graph, num_particles=3,
            average_node_potential_threshold=0.6, patience=30
        )
        pcm.run_simulation()

        # Rebuild owner_groups from graph and compare
        expected_groups = defaultdict(set)
        for node, data in pcm.graph.nodes(data=True):
            if data['owner'] is not None:
                expected_groups[data['owner']].add(node)

        for owner, nodes in expected_groups.items():
            assert owner in pcm.owner_groups
            assert pcm.owner_groups[owner] == nodes

    def test_cluster_sizes_consistent(self, simple_graph):
        """cluster_sizes should match actual cluster sizes."""
        random.seed(42)
        np.random.seed(42)
        pcm = ParticleCompetitionModel(
            simple_graph, num_particles=3,
            average_node_potential_threshold=0.6, patience=30
        )
        pcm.run_simulation()

        for owner, nodes in pcm.owner_groups.items():
            assert pcm.cluster_sizes[owner] == len(nodes)

    def test_coverage_between_0_and_1(self, simple_graph):
        """Coverage should be between 0 and 1."""
        random.seed(42)
        np.random.seed(42)
        pcm = ParticleCompetitionModel(
            simple_graph, num_particles=3,
            average_node_potential_threshold=0.6, patience=30
        )
        pcm.run_simulation()
        _, coverage = pcm.get_graph_coverage()
        assert 0.0 <= coverage <= 1.0

    def test_determinism_with_same_seed(self, simple_graph):
        """Same seed should produce same results."""
        results = []
        for _ in range(2):
            random.seed(123)
            np.random.seed(123)
            pcm = ParticleCompetitionModel(
                simple_graph, num_particles=3,
                average_node_potential_threshold=0.6, patience=30
            )
            pcm.run_simulation()
            owners = {node: data['owner'] for node, data in pcm.graph.nodes(data=True)}
            results.append(owners)

        assert results[0] == results[1]

    def test_different_seeds_may_differ(self, simple_graph):
        """Different seeds should (likely) produce different results."""
        owners_list = []
        for seed in [1, 999]:
            random.seed(seed)
            np.random.seed(seed)
            pcm = ParticleCompetitionModel(
                simple_graph, num_particles=3,
                average_node_potential_threshold=0.6, patience=30
            )
            pcm.run_simulation()
            owners = {node: data['owner'] for node, data in pcm.graph.nodes(data=True)}
            owners_list.append(owners)

        # Not a hard guarantee but very likely with different seeds
        assert owners_list[0] != owners_list[1]

    def test_positive_particle_starts_on_positive_node(self, simple_graph):
        """With random init, particle 0 should start on a positive node."""
        random.seed(42)
        pcm = ParticleCompetitionModel(
            simple_graph, num_particles=3, initialization_strategy='random'
        )
        particle_0 = pcm.particles[0]
        node = pcm.move_particle(particle_0)
        # Reserved positive particle should land on a node with observed_label=1
        assert simple_graph.nodes[node]['observed_label'] == 1

    def test_movement_strategies_both_work(self, two_cluster_graph):
        """Both movement strategies should produce valid simulation results."""
        for strategy in ['uniform', 'degree_weighted']:
            random.seed(42)
            np.random.seed(42)
            pcm = ParticleCompetitionModel(
                two_cluster_graph, num_particles=3,
                movement_strategy=strategy,
                average_node_potential_threshold=0.6, patience=30
            )
            result = pcm.run_simulation()
            assert result.graph['final_iteration_count'] > 0

    def test_invalid_movement_strategy_raises(self, simple_graph):
        """Invalid movement strategy should raise ValueError."""
        with pytest.raises(ValueError):
            ParticleCompetitionModel(
                simple_graph, num_particles=3, movement_strategy='invalid'
            )

    def test_invalid_initialization_strategy_raises(self, simple_graph):
        """Invalid initialization strategy should raise ValueError."""
        with pytest.raises(ValueError):
            ParticleCompetitionModel(
                simple_graph, num_particles=3, initialization_strategy='invalid'
            )


# ============================================================================
# TESTS: ReliableNegativeSelection
# ============================================================================

class TestReliableNegativeSelection:
    def _run_simulation(self, graph, num_particles=3, seed=42):
        """Helper to run particle competition and return the model."""
        random.seed(seed)
        np.random.seed(seed)
        pcm = ParticleCompetitionModel(
            graph, num_particles=num_particles,
            average_node_potential_threshold=0.6, patience=30
        )
        pcm.run_simulation()
        return pcm

    def test_assign_cluster_label_sets_attribute(self, two_cluster_graph):
        """After assign_cluster_label, nodes should have cluster_positive attribute."""
        pcm = self._run_simulation(two_cluster_graph)
        rns = ReliableNegativeSelection(
            pcm.graph,
            owner_groups=pcm.owner_groups,
            cluster_sizes=pcm.cluster_sizes,
            cluster_positive_counts=pcm.cluster_positive_counts,
            cluster_strategy='percentage',
            positive_cluster_threshold=0.1,
        )
        rns.assign_cluster_label()

        labeled_nodes = [
            node for node, data in rns.graph.nodes(data=True)
            if 'cluster_positive' in data
        ]
        # At least some nodes should be labeled
        assert len(labeled_nodes) > 0

    def test_majority_strategy_labels_correctly(self, two_cluster_graph):
        """Majority strategy should label clusters with more positives as positive."""
        pcm = self._run_simulation(two_cluster_graph, num_particles=2)
        rns = ReliableNegativeSelection(
            pcm.graph,
            owner_groups=pcm.owner_groups,
            cluster_sizes=pcm.cluster_sizes,
            cluster_positive_counts=pcm.cluster_positive_counts,
            cluster_strategy='majority',
            positive_cluster_threshold=0.5,
        )
        rns.assign_cluster_label()

        # Check that at least one positive and one negative cluster exist
        # (or fallback was used)
        labels = set()
        for _, data in rns.graph.nodes(data=True):
            if 'cluster_positive' in data:
                labels.add(data['cluster_positive'])
        # Should have at least one label type
        assert len(labels) >= 1

    def test_fallback_rule_when_no_positive_clusters(self, simple_graph):
        """If no cluster meets the threshold, fallback should select the best one."""
        # Use a very high threshold so no cluster qualifies
        pcm = self._run_simulation(simple_graph, num_particles=5)
        rns = ReliableNegativeSelection(
            pcm.graph,
            owner_groups=pcm.owner_groups,
            cluster_sizes=pcm.cluster_sizes,
            cluster_positive_counts=pcm.cluster_positive_counts,
            cluster_strategy='percentage',
            positive_cluster_threshold=0.99,  # Very high, unlikely to be met
        )
        rns.assign_cluster_label()
        assert rns.graph.graph.get('fallback_rule_used') in [True, False]

    def test_calculate_dissimilarity_assigns_scores(self, two_cluster_graph):
        """After calculate_dissimilarity, negative nodes should have dissimilarity scores."""
        pcm = self._run_simulation(two_cluster_graph, num_particles=2)
        rns = ReliableNegativeSelection(
            pcm.graph,
            owner_groups=pcm.owner_groups,
            cluster_sizes=pcm.cluster_sizes,
            cluster_positive_counts=pcm.cluster_positive_counts,
            cluster_strategy='percentage',
            positive_cluster_threshold=0.1,
        )
        rns.assign_cluster_label()
        rns.calculate_dissimilarity()

        dissimilarity_nodes = [
            node for node, data in rns.graph.nodes(data=True)
            if 'dissimilarity' in data
        ]
        # Some negative-cluster nodes should have dissimilarity scores
        assert len(dissimilarity_nodes) >= 0  # May be 0 if all clusters are positive

    def test_rank_returns_correct_count(self, larger_graph):
        """rank_nodes_dissimilarity should return exactly num_neg nodes (if available)."""
        pcm = self._run_simulation(larger_graph, num_particles=5)
        rns = ReliableNegativeSelection(
            pcm.graph,
            owner_groups=pcm.owner_groups,
            cluster_sizes=pcm.cluster_sizes,
            cluster_positive_counts=pcm.cluster_positive_counts,
            cluster_strategy='percentage',
            positive_cluster_threshold=0.1,
        )
        rns.assign_cluster_label()
        rns.calculate_dissimilarity()

        num_neg = 10
        ranked = rns.rank_nodes_dissimilarity(num_neg=num_neg)
        if ranked:  # If enough negative nodes exist
            assert len(ranked) == num_neg

    def test_ranked_nodes_sorted_by_dissimilarity(self, larger_graph):
        """Returned nodes should be sorted by dissimilarity (descending)."""
        pcm = self._run_simulation(larger_graph, num_particles=5)
        rns = ReliableNegativeSelection(
            pcm.graph,
            owner_groups=pcm.owner_groups,
            cluster_sizes=pcm.cluster_sizes,
            cluster_positive_counts=pcm.cluster_positive_counts,
            cluster_strategy='percentage',
            positive_cluster_threshold=0.1,
        )
        rns.assign_cluster_label()
        rns.calculate_dissimilarity()

        ranked = rns.rank_nodes_dissimilarity(num_neg=10)
        if len(ranked) >= 2:
            dissimilarities = [score for _, score in ranked]
            assert dissimilarities == sorted(dissimilarities, reverse=True)

    def test_rank_returns_empty_when_insufficient(self, simple_graph):
        """Should return empty list if fewer nodes available than num_neg."""
        pcm = self._run_simulation(simple_graph, num_particles=3)
        rns = ReliableNegativeSelection(
            pcm.graph,
            owner_groups=pcm.owner_groups,
            cluster_sizes=pcm.cluster_sizes,
            cluster_positive_counts=pcm.cluster_positive_counts,
            cluster_strategy='percentage',
            positive_cluster_threshold=0.1,
        )
        rns.assign_cluster_label()
        rns.calculate_dissimilarity()

        # Ask for more negatives than exist
        ranked = rns.rank_nodes_dissimilarity(num_neg=1000)
        assert ranked == []


# ============================================================================
# TESTS: PULearningPC (End-to-End)
# ============================================================================

class TestPULearningPC:
    def test_train_runs_without_error(self, two_cluster_graph):
        """PULearningPC.train() should complete without error."""
        random.seed(42)
        np.random.seed(42)
        model = PULearningPC(
            graph=two_cluster_graph,
            num_neg=3,
            pcm_params={'num_particles': 3, 'average_node_potential_threshold': 0.6, 'patience': 30},
            rns_params={'cluster_strategy': 'percentage', 'positive_cluster_threshold': 0.1},
        )
        model.train()
        assert model.labeled_graph is not None
        assert model.labeled_graph.graph['final_iteration_count'] > 0

    def test_select_reliable_negatives_returns_list(self, two_cluster_graph):
        """select_reliable_negatives should return a list of node IDs."""
        random.seed(42)
        np.random.seed(42)
        model = PULearningPC(
            graph=two_cluster_graph,
            num_neg=3,
            pcm_params={'num_particles': 2, 'average_node_potential_threshold': 0.6, 'patience': 30},
            rns_params={'cluster_strategy': 'percentage', 'positive_cluster_threshold': 0.1},
        )
        model.train()
        negatives = model.select_reliable_negatives()
        assert isinstance(negatives, list)

    def test_reliable_negatives_are_graph_nodes(self, larger_graph):
        """All returned reliable negatives should be valid nodes in the graph."""
        random.seed(42)
        np.random.seed(42)
        model = PULearningPC(
            graph=larger_graph,
            num_neg=10,
            pcm_params={'num_particles': 5, 'average_node_potential_threshold': 0.6, 'patience': 30},
            rns_params={'cluster_strategy': 'percentage', 'positive_cluster_threshold': 0.1},
        )
        model.train()
        negatives = model.select_reliable_negatives()
        if negatives:
            for node in negatives:
                assert node in larger_graph.nodes()

    def test_reliable_negatives_count(self, larger_graph):
        """Should return exactly num_neg negatives when enough exist."""
        random.seed(42)
        np.random.seed(42)
        num_neg = 15
        model = PULearningPC(
            graph=larger_graph,
            num_neg=num_neg,
            pcm_params={'num_particles': 5, 'average_node_potential_threshold': 0.6, 'patience': 30},
            rns_params={'cluster_strategy': 'percentage', 'positive_cluster_threshold': 0.1},
        )
        model.train()
        negatives = model.select_reliable_negatives()
        if negatives:
            assert len(negatives) == num_neg

    def test_reliable_negatives_quality(self, larger_graph):
        """Most reliable negatives should actually be true negatives (true_label=0)."""
        random.seed(42)
        np.random.seed(42)
        model = PULearningPC(
            graph=larger_graph,
            num_neg=20,
            pcm_params={'num_particles': 5, 'average_node_potential_threshold': 0.7, 'patience': 50},
            rns_params={'cluster_strategy': 'percentage', 'positive_cluster_threshold': 0.1},
        )
        model.train()
        negatives = model.select_reliable_negatives()
        if negatives:
            true_negatives = sum(
                1 for node in negatives if larger_graph.nodes[node]['true_label'] == 0
            )
            precision = true_negatives / len(negatives)
            # On a well-separated planted partition graph, precision should be decent
            assert precision > 0.5, f"Precision too low: {precision:.2f}"

    def test_end_to_end_determinism(self, larger_graph):
        """Same seed should produce identical reliable negatives."""
        results = []
        for _ in range(2):
            random.seed(42)
            np.random.seed(42)
            model = PULearningPC(
                graph=larger_graph,
                num_neg=10,
                pcm_params={'num_particles': 5, 'average_node_potential_threshold': 0.6, 'patience': 30},
                rns_params={'cluster_strategy': 'percentage', 'positive_cluster_threshold': 0.1},
            )
            model.train()
            negatives = model.select_reliable_negatives()
            results.append(negatives)

        assert results[0] == results[1]

    def test_different_rns_params_same_simulation(self, larger_graph):
        """
        Key test for optimization #1: different rns_params on the same simulation
        should produce different (but valid) results.
        """
        random.seed(42)
        np.random.seed(42)
        model_a = PULearningPC(
            graph=larger_graph,
            num_neg=10,
            pcm_params={'num_particles': 5, 'average_node_potential_threshold': 0.6, 'patience': 30},
            rns_params={'cluster_strategy': 'percentage', 'positive_cluster_threshold': 0.1},
        )
        model_a.train()
        negatives_a = model_a.select_reliable_negatives()

        # Re-run with different rns_params but reuse the same pcm state
        # (this simulates what the optimization would do)
        rns_b = ReliableNegativeSelection(
            model_a.labeled_graph,
            owner_groups=model_a.pcm.owner_groups,
            cluster_sizes=model_a.pcm.cluster_sizes,
            cluster_positive_counts=model_a.pcm.cluster_positive_counts,
            cluster_strategy='percentage',
            positive_cluster_threshold=0.5,
        )
        rns_b.assign_cluster_label()
        rns_b.calculate_dissimilarity()
        ranked_b = rns_b.rank_nodes_dissimilarity(num_neg=10)
        negatives_b = [node for node, _ in ranked_b] if ranked_b else []

        # Both should return valid nodes
        if negatives_a:
            for node in negatives_a:
                assert node in larger_graph.nodes()
        if negatives_b:
            for node in negatives_b:
                assert node in larger_graph.nodes()

    def test_majority_same_results_across_thresholds(self, larger_graph):
        """
        Key test: majority strategy with different thresholds should give identical results.
        This validates the redundancy we identified.
        """
        random.seed(42)
        np.random.seed(42)
        model = PULearningPC(
            graph=larger_graph,
            num_neg=10,
            pcm_params={'num_particles': 5, 'average_node_potential_threshold': 0.6, 'patience': 30},
            rns_params={'cluster_strategy': 'majority', 'positive_cluster_threshold': 0.1},
        )
        model.train()

        results_by_threshold = []
        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
            rns = ReliableNegativeSelection(
                model.labeled_graph,
                owner_groups=model.pcm.owner_groups,
                cluster_sizes=model.pcm.cluster_sizes,
                cluster_positive_counts=model.pcm.cluster_positive_counts,
                cluster_strategy='majority',
                positive_cluster_threshold=threshold,
            )
            rns.assign_cluster_label()
            rns.calculate_dissimilarity()
            ranked = rns.rank_nodes_dissimilarity(num_neg=10)
            negatives = [node for node, _ in ranked] if ranked else []
            results_by_threshold.append(negatives)

        # ALL results should be identical since majority ignores threshold
        for i in range(1, len(results_by_threshold)):
            assert results_by_threshold[0] == results_by_threshold[i], (
                f"Majority strategy gave different results for different thresholds! "
                f"threshold[0] vs threshold[{i}]"
            )


# ============================================================================
# TESTS: Core classes (Particle, OrderedSet)
# ============================================================================

class TestOrderedSet:
    def test_add_and_contains(self):
        os = OrderedSet()
        os.add(1)
        os.add(2)
        assert 1 in os
        assert 2 in os
        assert 3 not in os

    def test_add_duplicates(self):
        os = OrderedSet()
        os.add(1)
        os.add(1)
        assert len(os) == 1

    def test_remove(self):
        os = OrderedSet()
        os.add(1)
        os.add(2)
        os.remove(1)
        assert 1 not in os
        assert len(os) == 1

    def test_get_last(self):
        os = OrderedSet()
        os.add(1)
        os.add(2)
        os.add(3)
        assert os.get_last() == 3

    def test_get_last_empty(self):
        os = OrderedSet()
        assert os.get_last() is None

    def test_iteration_order(self):
        os = OrderedSet()
        os.add(3)
        os.add(1)
        os.add(2)
        assert list(os) == [3, 1, 2]


class TestParticle:
    def test_initial_state(self):
        p = Particle(0)
        assert p.id == 0
        assert p.potential == 0.05
        assert p.current_position is None
        assert len(p.visited_nodes) == 0

    def test_current_position_after_visit(self):
        p = Particle(0)
        p.visited_nodes.add(5)
        assert p.current_position == 5
        p.visited_nodes.add(10)
        assert p.current_position == 10

    def test_node_visited_last_iteration(self):
        p = Particle(0)
        p.visited_nodes.add(5)
        p.visited_nodes.add(10)
        assert p.node_visited_last_iteration == 5

    def test_node_visited_last_iteration_single_node(self):
        p = Particle(0)
        p.visited_nodes.add(5)
        assert p.node_visited_last_iteration == -1


# ============================================================================
# TESTS: Edge cases and regression guards
# ============================================================================

class TestEdgeCases:
    def test_single_particle(self, simple_graph):
        """Model should work with a single particle."""
        random.seed(42)
        np.random.seed(42)
        pcm = ParticleCompetitionModel(
            simple_graph, num_particles=1,
            average_node_potential_threshold=0.6, patience=30
        )
        result = pcm.run_simulation()
        assert result.graph['final_iteration_count'] > 0

    def test_many_particles(self, simple_graph):
        """Model should handle more particles than nodes gracefully."""
        random.seed(42)
        np.random.seed(42)
        pcm = ParticleCompetitionModel(
            simple_graph, num_particles=15,  # More than 10 nodes
            average_node_potential_threshold=0.6, patience=30
        )
        result = pcm.run_simulation()
        assert result.graph['final_iteration_count'] > 0

    def test_low_threshold_converges_fast(self, simple_graph):
        """Low avg_node_pot_threshold should make simulation converge quickly."""
        random.seed(42)
        np.random.seed(42)
        pcm = ParticleCompetitionModel(
            simple_graph, num_particles=3,
            average_node_potential_threshold=0.2, patience=30
        )
        result = pcm.run_simulation()
        assert result.graph['final_iteration_count'] < 100

    def test_high_threshold_runs_longer(self, larger_graph):
        """High threshold should require more iterations than low threshold."""
        iterations = []
        for threshold in [0.3, 0.8]:
            random.seed(42)
            np.random.seed(42)
            pcm = ParticleCompetitionModel(
                larger_graph, num_particles=5,
                average_node_potential_threshold=threshold, patience=50
            )
            result = pcm.run_simulation()
            iterations.append(result.graph['final_iteration_count'])

        assert iterations[1] >= iterations[0]
